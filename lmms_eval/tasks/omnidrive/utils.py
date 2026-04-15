# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from pathlib import Path

# Third-party
import yaml

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)

with open(Path(__file__).parent / "omnidrive.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def doc_to_visual_as_videos(doc):
    return [os.path.join(vision_cache_dir, path) for path in doc["video_paths"]]


def doc_to_target(doc, model_specific_target_kwargs=None):
    if model_specific_target_kwargs is None:
        model_specific_target_kwargs = {}
    return None


def process_results_omni(doc, results):
    return {
        "omni_gen": {
            "gt": doc["answer"],
            "pred": results[0],
        }
    }


def aggregate_omni_result(results):
    # Late imports: heavy dependencies, only needed for this metric
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    refs = {i: [res["gt"]] for i, res in enumerate(results)}
    hyps = {i: [res["pred"]] for i, res in enumerate(results)}

    total_scores = {}

    # BLEU
    bleu = Bleu(4)
    bleu_score, _ = bleu.compute_score(refs, hyps)
    for i in range(4):
        total_scores[f"Bleu_{i + 1}"] = bleu_score[i]
    total_scores["Bleu"] = sum(bleu_score) / 4

    # ROUGE_L
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(refs, hyps)
    total_scores["ROUGE_L"] = rouge_score

    # CIDEr
    cider = Cider()
    cider_score, _ = cider.compute_score(refs, hyps)
    total_scores["CIDEr"] = cider_score

    return total_scores


class AnswerFilter(Filter):

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_after_think_content(r) for r in resp]
            resp = [extract_final_boxed_content(r) for r in resp]
            filtered_resps.append(resp[0])
        return filtered_resps