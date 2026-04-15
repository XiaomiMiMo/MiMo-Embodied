# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
import re
from pathlib import Path

# Third-party
import yaml
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

QA_TEMPLATE = """
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.

Considering the progress shown in the video and my current observation in the last frame, what action should I take next in order to {}?

A. {}
B. {}
C. {}
D. {}
"""

with open(Path(__file__).parent / "egoplan.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

vision_cache_dir = yaml.safe_load("".join(safe_data))["img_root"]


def extract_after_think(text: str) -> str:
    """
    Extracts the part of the string that comes after </think>.
    If no <think>...</think> section exists, returns the original string.
    """
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def doc_to_visual_video(doc):
    qa_id = doc["sample_id"]
    visual_input = os.path.join(vision_cache_dir, qa_id + ".mp4")
    if not os.path.exists(visual_input):
        raise FileNotFoundError(f"Video not found: {visual_input}")
    return [visual_input]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    text_input = QA_TEMPLATE.format(
        doc["task_goal"],
        doc["choice_a"],
        doc["choice_b"],
        doc["choice_c"],
        doc["choice_d"],
    )
    return f"{pre_prompt}{text_input}{post_prompt}"


def process_results(doc, results):
    pred = results[0]
    extraction = extract_after_think(pred).lower()
    gold = doc["golden_choice_idx"].lower()
    acc = 1.0 if extraction == gold else 0.0
    return {"accuracy": acc}


def aggregate_results(results):
    total_acc = sum(results) / len(results)
    return total_acc