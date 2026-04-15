# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
import re
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from rouge import Rouge

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)

with open(Path(__file__).parent / "_default_boxed_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]
model_path = config["model_path"]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def doc_to_visual(doc):
    images = []
    image_paths = doc["image_paths"]
    for path in image_paths:
        path = os.path.join(vision_cache_dir, path)
        image = Image.open(path).convert("RGB")
        images.append(image)

    return images


def doc_to_target(doc, model_specific_target_kwargs=None):
    if model_specific_target_kwargs is None:
        model_specific_target_kwargs = {}
    return None


def process_test_results_for_submission(doc, results):
    metrics = ["acc", "rouge_1", "rouge_l", "semscore"]
    id = doc["id"]
    question = doc["question"]
    answer = results[0]

    frame_info = {
        "id": id,
        "question": question,
        "answer": answer,
        "gt": doc["answer"],
    }

    return {f"{metric}": frame_info for metric in metrics}


def idkb_acc(results):
    correct = 0
    for result in results:
        if sorted(result["answer"].replace(" ", "").lower()) == sorted(result["gt"].replace(" ", "").lower()):
            correct += 1
    return correct / len(results)


def _ensure_nonempty_for_rouge(text):
    """Normalize text to ensure ROUGE never receives an empty hypothesis/reference."""
    normalized = text or ""
    if not isinstance(normalized, str):
        normalized = str(normalized)
    normalized = normalized.strip()
    # Truncate to avoid extremely long inputs
    normalized = normalized[:1000]
    if not normalized:
        return "empty"
    # If content becomes empty after removing non-alphanumerics, fallback
    ascii_alnum_only = re.sub(r"[^A-Za-z0-9]+", " ", normalized).strip()
    if not ascii_alnum_only:
        return "empty"
    return normalized


def idkb_rouge(results, metric):
    rouge = Rouge(metrics=["rouge-1", "rouge-l"])
    preds = []
    golds = []

    for result in results:
        pred = _ensure_nonempty_for_rouge(result.get("answer"))
        gold = _ensure_nonempty_for_rouge(result.get("gt"))
        preds.append(pred)
        golds.append(gold)

    try:
        scores = rouge.get_scores(preds, golds, avg=True)
    except Exception:
        # As a last resort, replace any items that might still be problematic
        preds = ["empty" if not (p or "").strip() else p for p in preds]
        golds = ["empty" if not (g or "").strip() else g for g in golds]
        scores = rouge.get_scores(preds, golds, avg=True)

    return scores[metric]["f"] * 100


def idkb_rouge_1(results):
    return idkb_rouge(results, "rouge-1")


def idkb_rouge_l(results):
    return idkb_rouge(results, "rouge-l")


def idkb_semscore(results):
    # Late import: heavy dependency, only needed for this metric
    from sentence_transformers import SentenceTransformer, util

    preds = []
    golds = []
    for result in results:
        preds.append(result["answer"].strip())
        golds.append(result["gt"].strip())

    # TODO: Replace hardcoded path with a configurable model name or environment variable

    device = os.environ.get("SEMSCORE_DEVICE", "cuda:0")
    model = SentenceTransformer(model_path)
    model = model.to(device)

    combined_sentences = preds + golds
    embeddings = model.encode(combined_sentences, convert_to_tensor=True, device=device)

    ref_embeddings = embeddings[:len(preds)]
    gt_embeddings = embeddings[len(preds):]

    cosine_scores = util.cos_sim(ref_embeddings, gt_embeddings).diagonal()
    avg_similarity = cosine_scores.mean().item() * 100

    return avg_similarity


class AnswerFilter(Filter):

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_final_boxed_content(r) for r in resp]
            resp = [extract_after_think_content(r) for r in resp]
            filtered_resps.append(resp[0])

        return filtered_resps