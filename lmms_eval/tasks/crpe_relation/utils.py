# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

with open(Path(__file__).parent / "crpe_relation.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

vision_cache_dir = yaml.safe_load("".join(safe_data))["img_root"]


def crpe_doc_to_visual(doc):
    image_path = os.path.join(vision_cache_dir, doc["image_path"])
    image = Image.open(image_path)
    return [image]


def crpe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + lmms_eval_specific_kwargs.get("post_prompt")


def crpe_process_results(doc, results):
    prediction = results[0]
    final_answer = prediction.lower()
    gt_answer = doc["answer"].lower()

    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def crpe_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total