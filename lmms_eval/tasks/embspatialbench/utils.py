# Copyright 2025 Xiaomi Corporation.

from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

def embspatial_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]

def embspatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + lmms_eval_specific_kwargs.get("post_prompt")

def embspatial_process_results(doc, results):
    prediction = results[0]
    final_answer = prediction.lower()
    gt_answer = doc['answer_letter'].lower()
    
    if final_answer == gt_answer:
        return {"accuracy": 1.0} 
    else:
        return {"accuracy": 0.0} 
    

def embspatial_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 


