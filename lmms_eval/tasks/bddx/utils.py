# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from collections import defaultdict
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge as RougeCap

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)

with open(Path(__file__).parent / "bddx.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]


def _to_float(value, default=0.0):
    try:
        if isinstance(value, str):
            value = value.strip()
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def doc_to_visual(doc):
    images = []
    image_paths = doc["image_paths"]
    for path in image_paths:
        path = os.path.join(vision_cache_dir, path.replace(".jpg", ".png"))
        image = Image.open(path).convert("RGB")
        images.append(image)

    return images


def doc_to_target(doc, model_specific_target_kwargs=None):
    if model_specific_target_kwargs is None:
        model_specific_target_kwargs = {}
    return None


def process_test_results_for_submission(doc, results):
    metrics = ["caption", "rsme_angle", "rsme_speed"]
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


def bddx_caption(results):
    if not isinstance(results, list) or len(results) == 0:
        return {}

    gts = defaultdict(list)
    res = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        sample_id = item.get("id")
        pred = item.get("answer")
        gt = item.get("gt")
        if sample_id is None:
            continue
        if isinstance(pred, str) and pred.strip():
            res[sample_id] = [pred.strip()]
        if isinstance(gt, str) and gt.strip():
            gts[sample_id].append(gt.strip())

    valid_ids = [i for i in res.keys() if i in gts and len(gts[i]) > 0]
    gts = {i: gts[i] for i in valid_ids}
    res = {i: res[i] for i in valid_ids}

    if len(res) == 0 or len(gts) == 0:
        return {}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (RougeCap(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    result = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                result[m] = sc
        else:
            result[method] = score

    return {k: result[k] for k in ["Bleu_4", "ROUGE_L", "CIDEr"] if k in result}


class AnswerFilter(Filter):

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_final_boxed_content(r) for r in resp]
            resp = [extract_after_think_content(r) for r in resp]
            filtered_resps.append(resp[0])

        return filtered_resps