# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
import re
from pathlib import Path
# Third-party
import yaml
from loguru import logger as eval_logger
from transformers import pipeline

# Local
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

model_path = config["model_path"]


JUDGE_RULES = "[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"
pipe = pipeline("text-classification", model=model_path)

def extract_text_content(text, strict=False):
    """
    Extracts the content of the final \\text{} command in the given text.

    Args:
        text (str): The text containing \\text{} commands

    Returns:
        str or None: The content of the final \\text{} command, or None if no \\text{} command is found
    """
    text_matches = re.findall(r"\\text\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    if text_matches:
        return text_matches[-1]
    if strict:
        return ""
    return text


def lingoqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]

    return f"{pre_prompt}{question}{post_prompt}"


def lingoqa_doc_to_visual(doc):
    image_tokens = ["image_1", "image_2", "image_3", "image_4", "image_5"]
    return [doc[image_token].convert("RGB") for image_token in image_tokens]


def lingoqa_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


def lingo_process_results(doc, results):
    parsed_preds = []
    scores = []
    for pred in results:
        pred = extract_after_think_content(pred)
        pred = extract_final_boxed_content(pred)
        pred = extract_text_content(pred)
        llm_judge_prompt = JUDGE_RULES.format(
            question=doc["question"],
            answer=doc["answer"],
            prediction=pred,
        )
        llm_judge_score = pipe(llm_judge_prompt)[0]["score"]
        scores.append(llm_judge_score)
        parsed_preds.append(pred)

    lingo_judge_acc = {"answer": doc["answer"], "pred": parsed_preds, "score": scores}
    return {"lingo_judge_acc": lingo_judge_acc}


def lingo_aggregate_judge_results(results):
    total_score = 0
    for result in results:
        try:
            item_score = result["score"][0]
            if item_score > 0.5:
                total_score += 1
        except (KeyError, IndexError, TypeError):
            eval_logger.warning(f"Failed to extract score for {result}")
    return total_score / len(results)