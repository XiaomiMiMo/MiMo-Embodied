# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
import re
from pathlib import Path

# Third-party
import yaml
from PIL import Image

# Local
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

with open(Path(__file__).parent / "_default_boxed_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]


def cosmos_reason1_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["qa_pairs"]["question"], doc["qa_pairs"]["index2ans"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in choices.items()])
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def cosmos_reason1_doc_to_visual(doc):
    video_path = doc["video"]
    video_path = os.path.join(vision_cache_dir, video_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path: {video_path} does not exist, please check")

    return [video_path]


def cosmos_reason1_doc_to_target(doc, model_specific_target_kwargs=None):
    return doc["qa_pairs"]["answer"]


class MultiChoiceLetterRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
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
            option_letter_regex = re.compile(r"</think>\s*([A-Z])\.*")

            # Process each response
            filtered = []
            for resp in r:
                # Try to match the option letter at the start of the response
                match = option_letter_regex.search(resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    filtered.append(resp)

            filtered_resps.append(filtered[0])

        return filtered_resps