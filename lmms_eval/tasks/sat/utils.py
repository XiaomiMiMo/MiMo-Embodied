# Copyright 2025 Xiaomi Corporation.

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

def sat_doc_to_visual(doc):
    return [img.convert("RGB") for img in doc["images"]]


def sat_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + lmms_eval_specific_kwargs.get("post_prompt")


def sat_process_results(doc, results):
    prediction = results[0]
    final_answer = prediction.lower()
    gt_answer = doc["answer"].lower()

    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def sat_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
