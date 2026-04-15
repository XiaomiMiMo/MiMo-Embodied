# Copyright 2025 Xiaomi Corporation.

# Standard library
import json
import os
import re
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from loguru import logger as eval_logger

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

with open(Path(__file__).parent / "drivelm.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]


def replace_coordinates_with_normalized(texts):
    width = 1600
    height = 900
    pattern = r"<([^,]+),([^,]+),([\d.]+),([\d.]+)>"

    processed_texts = []

    for text in texts:
        def replace_match(match):
            obj_id = match.group(1)
            camera = match.group(2)
            x = float(match.group(3))
            y = float(match.group(4))

            norm_x = round(x / width, 2)
            norm_y = round(y / height, 2)

            return f"<{obj_id},{camera},{norm_x},{norm_y}>"

        processed_text = re.sub(pattern, replace_match, text)
        processed_texts.append(processed_text)

    return processed_texts


def replace_coordinates_with_unnormalized(texts):
    width = 1600
    height = 900
    pattern = r"<([^,]+),([^,]+),([\d.]+),([\d.]+)>"

    processed_texts = []

    for text in texts:
        def replace_match(match):
            obj_id = match.group(1)
            camera = match.group(2)
            x = float(match.group(3))
            y = float(match.group(4))

            norm_x = round(x * width, 2)
            norm_y = round(y * height, 2)

            return f"<{obj_id},{camera},{norm_x},{norm_y}>"

        processed_text = re.sub(pattern, replace_match, text)
        processed_texts.append(processed_text)

    return processed_texts


def drivelm_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]

    if lmms_eval_specific_kwargs.get("norm", False):
        question = replace_coordinates_with_normalized([question])[0]

    tag = doc["tag"]

    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    tag00_post_prompt = lmms_eval_specific_kwargs["tag00_post_prompt"]
    tag01_post_prompt = lmms_eval_specific_kwargs["tag01_post_prompt"]

    if tag == "0-0":        # mcq
        post_prompt = tag00_post_prompt
    elif tag == "0-1":      # judge
        post_prompt = tag01_post_prompt

    return f"{pre_prompt}{question}{post_prompt}"


def drivelm_doc_to_visual(doc):
    images = []
    image_paths = doc["image_paths"]
    for path in image_paths.split(", "):
        path = os.path.join(vision_cache_dir, path)
        image = Image.open(path).convert("RGB")
        images.append(image)

    return images


def drivelm_doc_to_target(doc, model_specific_target_kwargs):
    return None


def extract_final_boxed_content_for_drivelm(text):
    """
    Extracts the content of the final \\?boxed{} command in the given text.

    Args:
        text (str): The text containing \\?boxed{} commands

    Returns:
        str or None: The content of the final \\?boxed{} command, or None if no \\boxed{} command is found
    """
    boxed_matches = re.findall(r"\\?boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    if boxed_matches:
        return boxed_matches[-1]
    else:
        return None


def drivelm_process_test_results_for_submission(doc, results):
    id = doc["id"]
    tag = doc["tag"]
    question = doc["question"]
    answer = results[0]

    if tag == "0-0":
        if answer not in ["A", "B", "C", "D"]:
            answer = extract_final_boxed_content_for_drivelm(answer)
            eval_logger.error(
                f"tag: {tag}, question: {question}, answer: {answer}"
            )

    frame_info = {
        "id": id,
        "question": question,
        "answer": answer,
    }

    return {"submission": frame_info}


def drivelm_test_aggregate_results_for_submission(results, args):
    file = generate_submission_file("drivelm_submission.json", args)
    eval_logger.info(f"Results count: {len(results)}")

    submission_content = {
        "method": "llm",
        "team": "eee",
        "authors": ["mike"],
        "email": "123456789@qq.com",
        "institution": "no",
        "country": "China",
        "results": results,
    }

    with open(file, "w") as f:
        json.dump(submission_content, f, indent=4)

    eval_logger.info(f"Submission file saved to {file}")


class DriveLmFilter(Filter):
    def __init__(self, unnorm=False, **kwargs):
        super().__init__(self, **kwargs)
        self.unnorm = unnorm

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_after_think_content(r) for r in resp]
            resp = [extract_final_boxed_content(r) for r in resp]
            resp = resp[0]
            if self.unnorm:
                resp = replace_coordinates_with_unnormalized([resp])[0]

            filtered_resps.append(resp)

        return filtered_resps