# Copyright 2025 Xiaomi Corporation.

# Standard library
import json
import os
from pathlib import Path
from typing import List

# Third-party
import yaml
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image

# Local
from lmms_eval.api.filter import Filter
from lmms_eval.tasks._task_utils.eval_utils import (
    extract_after_think_content,
    extract_final_boxed_content,
)

with open(Path(__file__).parent / "maplm.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]

with open(Path(__file__).parent / "problems.json") as f:
    frames = json.load(f)

with open(Path(__file__).parent / "pid_splits.json") as f:
    pid_splits = json.load(f)

frame_ids = pid_splits["test"]


def retrieve_completion(question, conversations):
    if question == "How many lanes in current road?":
        question = "How many lanes on the current road?"

    if question == "Discribe the lane attribute in current road.":
        question = "Describe the lane attribute in the current road."

    for i, conversation in enumerate(conversations):
        if question in conversation["question"]:
            return conversation["answer"]


def completion_to_answer(completion, choices):
    if not completion.endswith("."):
        completion = completion + "."
    for i, choice in enumerate(choices):
        if choice.lower() in completion.lower():
            return i


def output_format(output):
    newoutput = {}
    for item in output:
        if item["id"] not in newoutput:
            newoutput[item["id"]] = [item]
        else:
            newoutput[item["id"]].extend([item])
    return newoutput


def acc_counter():
    return {
        "total": 0,
        "correct": 0,
    }


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
    MAPLM_METRICS = ["FRM", "QNS", "des", "scn", "qlt", "lan", "int"]
    id = doc["frame_id"]
    question = doc["question"]
    answer = results[0]

    frame_info = {
        "id": id,
        "question": question,
        "answer": answer,
    }

    return {f"maplm_{metric}": frame_info for metric in MAPLM_METRICS}


def maplm_aggregate_results(results, metric):
    model_output = output_format(results)
    final_results = dict(
        QNS=acc_counter(),
        FRM=acc_counter(),
    )

    for i, frame_id in enumerate(frame_ids):
        frame = frames[frame_id]
        qas = frame["qa"]
        corrects = []

        model_frame_output = model_output[frame_id]

        for j, qa in enumerate(qas):
            if qa["task"] == "closed choice":
                question = qa["question"]
                choices: List[str] = qa["choices"]
                true_answer: int = qa["answer"]

                completion = retrieve_completion(question, model_frame_output)
                pred_answer = completion_to_answer(completion, choices)

                if question not in final_results:
                    final_results[question] = acc_counter()

                correct = bool(pred_answer == true_answer)
                corrects.append(correct)
                final_results[question]["total"] += 1
                final_results[question]["correct"] += int(correct)
                final_results["QNS"]["total"] += 1
                final_results["QNS"]["correct"] += int(correct)
            else:
                question = qa["question"]
                true_answer: str = qa["answer"]

                if question not in final_results:
                    final_results[question] = acc_counter()

                completion = retrieve_completion(question, model_frame_output)

                final_results[question]["total"] += 1
                final_results[question]["correct"] += sentence_bleu(
                    [true_answer.split()],
                    completion.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                )

        final_results["FRM"]["total"] += 1
        final_results["FRM"]["correct"] += int(all(corrects))

    return final_results[metric]["correct"] / final_results[metric]["total"] * 100


def maplm_frm(results):
    return maplm_aggregate_results(results, "FRM")


def maplm_qns(results):
    return maplm_aggregate_results(results, "QNS")


def maplm_des(results):
    return maplm_aggregate_results(results, "Describe the lane attribute in the current road.")


def maplm_scn(results):
    return maplm_aggregate_results(results, "What kind of road scene is it in the images?")


def maplm_qlt(results):
    return maplm_aggregate_results(results, "What is the point cloud data quality in current road area of this image?")


def maplm_lan(results):
    return maplm_aggregate_results(results, "How many lanes on the current road?")


def maplm_int(results):
    return maplm_aggregate_results(results, "Is there any road cross, intersection or lane change zone in the main road?")


class AnswerFilter(Filter):

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp = [extract_after_think_content(r) for r in resp]
            resp = [extract_final_boxed_content(r) for r in resp]
            filtered_resps.append(resp[0])

        return filtered_resps