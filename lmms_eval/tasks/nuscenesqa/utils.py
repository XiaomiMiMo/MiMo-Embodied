# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from pathlib import Path

# Third-party
import yaml
from PIL import Image

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)

with open(Path(__file__).parent / "nuscenesqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]


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


def process_test_results_for_submission_nuscenesqa(doc, results):
    id = doc["id"]
    question = doc["question"]
    answer = results[0]

    frame_info = {
        "id": id,
        "question": question,
        "answer": answer,
        "gt": doc["answer"],
        "type": doc["type"],
        "num_hop": doc["num_hop"],
    }

    return {"accuracy": frame_info}


def nuscenesqa_aggregate_results(results):
    dic = {"exist": 0, "count": 0, "object": 0, "status": 0, "comparison": 0}
    dic_all = {"exist": 0, "count": 0, "object": 0, "status": 0, "comparison": 0}

    for result in results:
        dic_all[result["type"]] += 1
        if result["gt"].lower() == result["answer"].lower().replace(".", ""):
            dic[result["type"]] += 1

    ratio_dict = {}
    for key in dic:
        if key in dic_all and dic_all[key] != 0:
            ratio_dict[key] = dic[key] / dic_all[key]
        else:
            ratio_dict[key] = None

    return ratio_dict


class AnswerFilter(Filter):

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_final_boxed_content(r) for r in resp]
            resp = [extract_after_think_content(r) for r in resp]
            filtered_resps.append(resp[0])

        return filtered_resps