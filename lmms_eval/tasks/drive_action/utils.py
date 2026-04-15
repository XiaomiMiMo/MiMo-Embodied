# Copyright 2025 Xiaomi Corporation.

# Standard library
import ast
import re
from collections import defaultdict

# Local
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content

DRIVE_ACTION_METRICS = ["Vision", "Language", "Action", "Overall"]

def exact_match(pred, gt):
    """Brought from MMStar"""
    answer = gt.lower().strip().replace("\n", " ")
    predict = pred.lower().strip().replace("\n", " ")
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0


def drive_action_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    content = doc["content"]
    question = content["question"]
    choices = content["options"]
    if isinstance(choices, str):
        choices = ast.literal_eval(choices)
    len_choices = len(choices)
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]

    choices_str = "\n".join([f"{option}. {choice}" for option, choice in choices.items()])
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def drive_action_doc_to_visual(doc):
    image_tokens = ["image_0", "image_1", "image_2"]
    return [doc[image].convert("RGB") for image in image_tokens]


def drive_action_doc_to_target(doc, model_specific_target_kwargs):
    return doc["content"]["answer"]


def drive_action_process_results(doc, results):
    preds = []
    scores = []
    for pred in results:
        answer = doc["content"]["answer"]
        score = exact_match(pred, answer)
        scores.append(score)
        preds.append(pred)

    drive_action_acc = {
        "question_slice_id": doc["question_slice_id"],
        "qa_l0": doc["qa_l0"],
        "qa_l1": doc["qa_l1"],
        "answer": doc["content"]["answer"],
        "pred": preds,
        "score": scores,
    }
    return {f"drive_action_{metric}_acc": drive_action_acc for metric in DRIVE_ACTION_METRICS}


def drive_action_aggregate_results(results, metric):
    if metric in ["Overall"]:
        overall_acc = sum([sample["score"][0] for sample in results]) / len(results)
        overall_acc = round(overall_acc, 5)
        return overall_acc

    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["qa_l0"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        subset_acc = sum([sample["score"][0] for sample in sub_eval_samples]) / len(sub_eval_samples)
        subset_metric_dict = {
            "num": len(sub_eval_samples),
            "acc": round(subset_acc, 5),
        }
        evaluation_result[subset] = subset_metric_dict
    return evaluation_result[metric]["acc"]


def drive_action_vision_acc(results):
    return drive_action_aggregate_results(results, "Vision")


def drive_action_language_acc(results):
    return drive_action_aggregate_results(results, "Language")


def drive_action_action_acc(results):
    return drive_action_aggregate_results(results, "Action")


def drive_action_overall_acc(results):
    return drive_action_aggregate_results(results, "Overall")


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            # Regex to directly extract the option letter from the model response
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            # Process each response
            filtered = []
            for resp in r:
                match = option_letter_regex.match(resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    filtered.append(resp)

            filtered_resps.append(filtered[0])

        return filtered_resps


class MultiChoiceBoxedRegexFilter(MultiChoiceRegexFilter):
    def apply(self, resps, docs):
        resps = [[extract_final_boxed_content(r)] for resp in resps for r in resp]
        filtered_resps = super().apply(resps, docs)
        return filtered_resps