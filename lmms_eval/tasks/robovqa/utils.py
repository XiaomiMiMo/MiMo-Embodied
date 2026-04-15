# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
import re
from pathlib import Path

# Third-party
import decord
import numpy as np
import yaml
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from lmms_eval.tasks._task_utils.eval_utils import AfterThinkFilter

decord.bridge.set_bridge("torch")

with open(Path(__file__).parent / "robovqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

vision_cache_dir = yaml.safe_load("".join(safe_data))["img_root"]

TAGS_RE = r"(</?[\w:]+>)"

# Mapping of abbreviated question forms to full natural-language forms
_QUESTION_REPLACEMENTS = [
    (". Q: immediate next step", ". what is the immediate next step"),
    (". Q: next 5 step", ". what are the next 5 step"),
    (". Q: satisfied", ". is it satisfied"),
    (". Q: possible right now", ". is it possible right now"),
    (" Q: immediate next step", ". what is the immediate next step"),
    (" Q: next 5 step", ". what are the next 5 step"),
    (" Q: satisfied", ". is it satisfied"),
    (" Q: possible right now", ". is it possible right now"),
]


def robovqa_doc_to_visual(doc, num_frames=8):
    video_filename = doc["video"]
    video_absolute_path = os.path.join(vision_cache_dir, video_filename)

    video_reader = decord.VideoReader(video_absolute_path, ctx=decord.cpu(0))
    total_frames = len(video_reader)

    if total_frames < num_frames:
        indices = np.arange(total_frames).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    batch = video_reader.get_batch(indices)

    if hasattr(batch, "asnumpy"):
        frames_np = batch.asnumpy()
    elif hasattr(batch, "cpu") and hasattr(batch, "numpy"):
        frames_np = batch.cpu().numpy()
    else:
        frames_np = np.asarray(batch)

    if frames_np.ndim != 4:
        raise ValueError(f"Unexpected frames batch shape: {frames_np.shape}")

    if frames_np.shape[1] in (1, 3, 4) and frames_np.shape[-1] not in (1, 3, 4):
        frames_np = frames_np.transpose(0, 2, 3, 1)

    if frames_np.dtype != np.uint8:
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frames_np = frames_np.clip(0, 255).astype(np.uint8)

    images = [Image.fromarray(frame) for frame in frames_np]

    return images


def robovqa_doc_to_text_mimo(doc: dict) -> str:
    full_text = doc.get("text", "")
    parts = re.split(r"<PRED>A:", full_text, maxsplit=1, flags=re.IGNORECASE)
    question_text = parts[0]
    cleaned_question_raw = re.sub(TAGS_RE, "", question_text)
    question = " ".join(cleaned_question_raw.strip().split())

    for old, new in _QUESTION_REPLACEMENTS:
        question = question.replace(old, new)
    question = question.replace("Q: ", "")

    if ". is it" in question:
        question = question + " Please answer yes or no. "

    return question


def robovqa_doc_to_target(doc: dict) -> str:
    full_text = doc.get("text", "")
    parts = re.split(r"<PRED>A:", full_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        return ""
    answer_part = parts[1]
    exact_answer_text = answer_part.split("</PRED>", 1)[0]
    cleaned_answer = re.sub(TAGS_RE, "", exact_answer_text)
    ans = " ".join(cleaned_answer.strip().split())
    return ans


def get_bleu_score(prediction, target):
    candidate = prediction.split(" ")
    reference = [target.split(" ")]

    if target is None:
        return 0, 0, 0, 0, 0

    ref_len = len(reference[0])

    if ref_len <= 1:
        weights_list = [
            (1.00, 0.00, 0.00, 0.00),
            (1.00, 0.00, 0.00, 0.00),
            (1.00, 0.00, 0.00, 0.00),
            (1.00, 0.00, 0.00, 0.00),
        ]
    elif ref_len == 2:
        weights_list = [
            (1.00, 0.00, 0.00, 0.00),
            (0.50, 0.50, 0.00, 0.00),
            (0.50, 0.50, 0.00, 0.00),
            (0.50, 0.50, 0.00, 0.00),
        ]
    elif ref_len == 3:
        weights_list = [
            (1.00, 0.00, 0.00, 0.00),
            (0.50, 0.50, 0.00, 0.00),
            (0.33, 0.33, 0.33, 0.00),
            (0.33, 0.33, 0.33, 0.00),
        ]
    else:
        weights_list = [
            (1.00, 0.00, 0.00, 0.00),
            (0.50, 0.50, 0.00, 0.00),
            (0.33, 0.33, 0.33, 0.00),
            (0.25, 0.25, 0.25, 0.25),
        ]

    bleu_scores = [
        sentence_bleu(reference, candidate, weights=w)
        for w in weights_list
    ]

    score = sum(bleu_scores) / 4
    return score, bleu_scores[0], bleu_scores[1], bleu_scores[2], bleu_scores[3]


def robovqa_process_results(doc: dict, results: list) -> dict:
    gold_answer = robovqa_doc_to_target(doc)
    prediction = results[0].strip()
    prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL | re.MULTILINE)
    prediction = re.sub(r"<[^>]+>", "", prediction)
    prediction = re.sub(r"\n+", " ", prediction)
    prediction = prediction.strip()
    pred = prediction.replace("\n", "").lower()
    gt = gold_answer.replace("\n", "").lower()

    if gt in ["yes", "no"]:
        pred = re.sub(r"\b\w*yes\w*\b", "yes", pred)
        pred = re.sub(r"\b\w*no\w*\b", "no", pred)

    score, bleu1, bleu2, bleu3, bleu4 = get_bleu_score(pred, gt)
    return {
        "robovqa_score": score,
        "robovqa_bleu1": bleu1,
        "robovqa_bleu2": bleu2,
        "robovqa_bleu3": bleu3,
        "robovqa_bleu4": bleu4,
    }


def robovqa_aggregate_results(items):
    if not items:
        return 0.0
    return np.mean(items)