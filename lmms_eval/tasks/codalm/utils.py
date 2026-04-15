# Copyright 2025 Xiaomi Corporation.

# Standard library
import json
import os
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from loguru import logger as eval_logger

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

with open(Path(__file__).parent / "_default_boxed_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
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


def process_test_results_for_submission(doc, results):
    id = doc["id"]
    question = doc["question"]
    answer = results[0]

    frame_info = {
        "id": id,
        "question": question,
        "answer": answer,
        "image_paths": doc["image_paths"],
    }

    return {"submission": frame_info}


def test_aggregate_results_for_submission(results, args):
    file = generate_submission_file("submission.json", args)
    eval_logger.info(f"Results count: {len(results)}")

    submission_content = {"results": results}

    with open(file, "w") as f:
        json.dump(submission_content, f, indent=4)

    eval_logger.info(f"Submission file saved to {file}")


class AnswerFilter(Filter):

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_after_think_content(r) for r in resp]
            resp = [extract_final_boxed_content(r) for r in resp]
            filtered_resps.append(resp[0])

        return filtered_resps