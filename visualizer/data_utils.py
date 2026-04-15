import ast
import streamlit as st
import os
import glob
import datasets
import json
import io
import base64
from PIL import Image
import re
from collections import defaultdict
import task_utils

from pydantic import BaseModel
from collections import namedtuple
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, Literal
from enum import Enum

import task_utils
from task_utils.media_utils import load_image_from_bytes


LMMS_EVAL_DATA_HOME = os.getenv("LMMS_EVAL_DATA_HOME", None)
assert LMMS_EVAL_DATA_HOME is not None, "Environment variable LMMS_EVAL_DATA_HOME is not set"

class DATASET_TYPE(Enum):
    HF = "hf"
    NONE = "none"


Filter = Union[str, Callable, Tuple[Union[str, Callable], ...]]

def apply(data: dict, filter: Filter, default: Any = None) -> Any:
    if not isinstance(filter, tuple):
        filter = (filter,)
    result = data
    for f in filter:
        if isinstance(f, str):
            result = result.get(f)
        else:
            try:
                result = f(result)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                result = None
        if result is None:
            return default
    return result


class Metric(BaseModel):
    name: str
    factor: float = 100
    higher_is_better: bool = True
    suffix: Optional[str] = None

class RawData(BaseModel):
    type: DATASET_TYPE = DATASET_TYPE.HF
    hf_path: Optional[str] = None
    hf_name: Optional[str] = None
    hf_split: Optional[str] = None

class Task(BaseModel):
    name: str
    raw_data: Optional[RawData] = None
    metric: Optional[Metric | List[Metric]] = None
    media_fn: Optional[Filter] = None
    gt_anno_fn: Optional[Filter] = None
    anno_fn: Optional[Filter] = None
    score_fn: Optional[Filter] = None
    tags_fn: Optional[Filter | List[Filter]] = None
    input_fn: Optional[Filter] = None
    answer_fn: Optional[Filter] = None
    output_fn: Optional[Filter] = None
    support_score: Optional[bool] = None

class Group(BaseModel):
    name: str = "default"
    raw_data: Optional[RawData] = None
    metric: Optional[Metric | List[Metric]] = None
    media_fn: Optional[Filter] = None
    gt_anno_fn: Optional[Filter] = None
    anno_fn: Optional[Filter] = None
    score_fn: Optional[Filter] = None
    tags_fn: Optional[Filter | List[Filter]] = None
    input_fn: Optional[Filter] = None
    answer_fn: Optional[Filter] = None
    output_fn: Optional[Filter] = None
    support_score: Optional[bool] = None
    tasks: List[Task]

class Dataset(BaseModel):
    name: str
    raw_data: Optional[RawData] = None
    metric: Optional[Metric | List[Metric]] = None
    media_fn: Optional[Filter] = None
    gt_anno_fn: Optional[Filter] = None
    anno_fn: Optional[Filter] = None
    score_fn: Optional[Filter] = None
    tags_fn: Optional[Filter | List[Filter]] = None
    input_fn: Optional[Filter] = None
    answer_fn: Optional[Filter] = None
    output_fn: Optional[Filter] = None
    support_score: Optional[bool] = None
    groups: List[Group]

_default_dataset = Dataset(name="default", media_fn="image", input_fn="input", answer_fn="target", output_fn=lambda x: x["resps"][0], groups=[])

def mmmu_image_fn(item):
    images = []
    for i in range(1, 8):
        if item.get(f"image_{i}") is not None:
            images.append(item[f"image_{i}"])
    return images

def blink_image_fn(item):
    imgs = []
    for i in range(1, 5):
        img = item.get(f"image_{i}")
        if img is None:
            break
        imgs.append(img)
    return imgs


def android_control_gt_anno_fn(item):
    if "gt_coord" not in item:
        return None
    x, y = item["gt_coord"]
    width, height = item["image_size"]
    gt_annotations = [{
        "type": "point",
        "value": (x, y)
    },{
        "type": "circle",
        "value": {
            "center": (x, y),
            "radius": width * 0.14
        }
    }]
    return gt_annotations

def android_control_anno_fn(item):
    if "pred_coord" not in item:
        return None
    x, y = item["pred_coord"]
    annotations = [{
        "type": "point",
        "value": (x, y)
    }]
    return annotations

def get_answer_type(ans):
    try:
        ans_float = float(ans.strip("%"))
        return 'numerical'
    except ValueError:
        if ans.lower().strip('.') in ["yes", "no"]:
            return 'yes/no'
        else:
            return 'text'

def get_media(media_fn, item):
    if media_fn is None:
        return None
    media = apply(item, media_fn)
    if media is None:
        return None
    if isinstance(media, list):
        return media
    return [media]


def get_tags(tags_fn, sample):
    if tags_fn is None:
        return None
    if not isinstance(tags_fn, list):
        tags_fn = [tags_fn]
    doc = sample.get("doc", {})
    tags = []
    for _tags_fn in tags_fn:
        _tags = apply(doc, _tags_fn)
        if _tags is not None:
            if isinstance(_tags, list):
                tags.extend(_tags)
            else:
                tags.append(_tags)
    return tags


def get_cii_bench_tags(doc, key):
    if key in doc:
        try:
            _tags = ast.literal_eval(doc[key])['choices']
        except:
            _tags = [doc[key]]
        return _tags
    return []

def make_list(x):
    return [x]






DATASETS = [
    Dataset(name="mmmu_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/MMMU"), hf_split="validation"),
        media_fn=mmmu_image_fn,
        tags_fn="subfield", 
        groups=[
            Group(
                name="default",
                metric=Metric(name="mmmu_acc", factor=100, higher_is_better=True),
                score_fn=("mmmu_acc", task_utils.mmmu.eval_mmmu_sample_score),
                tasks=[
                    Task(name="mmmu_val"), 
                    Task(name="mmmu_val_boxed_gpt", metric=Metric(name="mmmu_judge_acc", factor=100, higher_is_better=True), score_fn=("mmmu_judge_acc", "score", lambda x: int(x[0]))),
                    Task(name="mmmu_val_reasoning", metric=Metric(name="mmmu_judge_acc", factor=100, higher_is_better=True), score_fn=("mmmu_judge_acc", "score", lambda x: int(x[0]))),
                ]
            )
        ]
    ), 
    Dataset(name="cmmmu_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/CMMMU"), hf_split="val"),
        media_fn=mmmu_image_fn,
        groups=[
            Group(
                name="default",
                metric=Metric(name="cmmmu_acc", factor=100, higher_is_better=True),
                score_fn=("cmmmu_acc", task_utils.cmmmu.eval_cmmmu_sample_score),
                tasks=[
                    Task(name="cmmmu_val"), 
                    Task(name="cmmmu_val_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="mmmu_pro",
        support_score=True,
        metric=Metric(name="mmmu_acc", factor=100, higher_is_better=True),
        score_fn=("mmmu_acc", task_utils.mmmu_pro.eval_mmmu_pro_sample_score),
        groups=[
            Group(
                name="standard",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "MMMU/MMMU_Pro"), hf_name="standard (10 options)", hf_split="test"),
                media_fn=mmmu_image_fn,
                tasks=[
                    Task(name="mmmu_pro_standard"),
                    Task(name="mmmu_pro_standard_boxed"),
                ]
            ), 
            Group(
                name="vision",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "MMMU/MMMU_Pro"), hf_name="vision", hf_split="test"),
                tasks=[
                    Task(name="mmmu_pro_vision"),
                    Task(name="mmmu_pro_vision_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="ocrbench",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "echo840/OCRBench"), hf_split="test"),
        metric=Metric(name="ocrbench_accuracy", factor=100, higher_is_better=True),
        score_fn=("ocrbench_accuracy", "score"),
        tags_fn=["question_type", "dataset"], 
        groups=[
            Group(
                tasks=[
                    Task(name="ocrbench"),
                    Task(name="ocrbench_boxed_gpt"),
                ]
            )
        ]
    ),
    Dataset(name="docvqa_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/DocVQA"), hf_name="DocVQA", hf_split="validation"),
        tags_fn=[
            ("answers", lambda x: x[0], get_answer_type),
            # "question_types",
        ], 
        groups=[
            Group(
                name="default",
                metric=Metric(name="anls_strip_period", factor=100, higher_is_better=True, suffix="sp"), 
                score_fn="anls_strip_period",
                tasks=[
                    Task(name="docvqa_val"),
                    Task(name="docvqa_val_boxed_gpt", metric=Metric(name="gpt_eval_score", factor=100, higher_is_better=True), score_fn=("gpt_eval_score", "score")),
                ]
            )
        ]
    ), 
    Dataset(name="ai2d",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/ai2d"), hf_split="test"),
        metric=Metric(name="exact_match", factor=100, higher_is_better=True),
        score_fn="exact_match",
        groups=[
            Group(
                tasks=[
                    Task(name="ai2d"),
                    Task(name="ai2d_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="mmstar",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "Lin-Chen/MMStar"), hf_split="val"),
        metric=Metric(name="average", factor=100, higher_is_better=True),
        score_fn=("average", "score"),
        groups=[
            Group(
                tasks=[
                    Task(name="mmstar"),
                ]
            )
        ]
    ), 
    Dataset(name="chartqa",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/ChartQA"), hf_split="test"),
        metric=Metric(name="relaxed_overall", factor=100, higher_is_better=True),
        tags_fn=[
            ("answer", get_answer_type),
            "type"
        ],
        groups=[
            Group(
                name="default",
                score_fn="relaxed_overall",
                tasks=[
                    Task(name="chartqa"),
                    Task(name="chartqa_boxed_gpt", score_fn=("relaxed_overall", "score")),
                ]
            )
        ]
    ), 
    Dataset(name="infovqa_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/DocVQA"), hf_name="InfographicVQA", hf_split="validation"),
        metric=Metric(name="anls_strip_period", factor=100, higher_is_better=True, suffix="sp"),
        score_fn="anls_strip_period",
        tags_fn=("answers", lambda x: x[0], get_answer_type),
        groups=[
            Group(
                tasks=[
                    Task(name="infovqa_val"),
                    Task(name="infovqa_val_boxed_gpt", metric=Metric(name="gpt_eval_score", factor=100, higher_is_better=True), score_fn=("gpt_eval_score", "score")),
                ]
            )
        ]
    ),
    Dataset(name="seedbench", 
        support_score=True, 
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/SEED-Bench"), hf_split="test"),
        metric=Metric(name="seed_all", factor=100, higher_is_better=True),
        score_fn=("seed_all", lambda x: int(x['pred'] == x['answer'])),
        groups=[
            Group(
                tasks=[
                    Task(name="seedbench"),
                    Task(name="seedbench_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="seedbench_2", 
        support_score=True, 
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/SEED-Bench-2"), hf_split="test"),
        metric=Metric(name="seed_all", factor=100, higher_is_better=True),
        score_fn=("seed_all", task_utils.seedbench_2.eval_seedbench_2_sample_score),
        groups=[
            Group(
                tasks=[
                    Task(name="seedbench_2"),
                    Task(name="seedbench_2_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="seedbench_2_plus", 
        support_score=True, 
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "doolayer/SEED-Bench-2-Plus"), hf_split="test"),
        metric=Metric(name="seedbench_2_plus_all", factor=100, higher_is_better=True),
        score_fn=("seedbench_2_plus_all", task_utils.seedbench_2.eval_seedbench_2_sample_score),
        groups=[
            Group(
                tasks=[
                    Task(name="seedbench_2_plus"),
                    Task(name="seedbench_2_plus_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="mmbench_dev",
        support_score=True,
        metric=Metric(name="gpt_eval_score", factor=1, higher_is_better=True),
        score_fn=("gpt_eval_score", "score"),
        groups=[
            Group(
                name="en",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/MMBench"), hf_name="en", hf_split="dev"),
                tasks=[
                    Task(name="mmbench_en_dev"),
                    Task(name="mmbench_en_dev_boxed"),
                ]
            ), 
            Group(
                name="cn",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/MMBench"), hf_name="cn", hf_split="dev"),
                tasks=[
                    Task(name="mmbench_cn_dev"),
                    Task(name="mmbench_cn_dev_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="mmerealworld",
        support_score=True,
        metric=Metric(name="mme_realworld_score", factor=100, higher_is_better=True),
        media_fn=("bytes", load_image_from_bytes),
        score_fn=("mme_realworld_score", lambda x: int(x.get("answer", None) == x.get("pred_answer", None))),
        groups=[
            Group(
                name="default",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "yifanzhang114/MME-RealWorld-Lmms-eval"), hf_split="train"),
                tasks=[
                    Task(name="mmerealworld"),
                    Task(name="mmerealworld_boxed"),
                ]
            ), 
            Group(
                name="cn",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "yifanzhang114/MME-RealWorld-CN-Lmms-eval"), hf_split="train"),
                tasks=[
                    Task(name="mmerealworld_cn"),
                    Task(name="mmerealworld_cn_boxed"),
                ]
            ), 
            Group(
                name="lite",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "yifanzhang114/MME-RealWorld-lite-lmms-eval"), hf_split="train"),
                tasks=[
                    Task(name="mmerealworld_lite"),
                    Task(name="mmerealworld_lite_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="vibe_eval",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "RekaAI/VibeEval"), hf_split="test"),
        metric=Metric(name="all", factor=1, higher_is_better=True),
        score_fn=("all", "score", lambda x: x/5),
        groups=[
            Group(
                tasks=[
                    Task(name="vibe_eval"),
                    Task(name="vibe_eval_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="ocrbench_v2",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "ling99/OCRBench_v2"), hf_split="test"),
        metric=Metric(name="ocrbench_v2_accuracy", factor=100, higher_is_better=True),
        score_fn=("ocrbench_v2_accuracy", "score"),
        answer_fn=("doc", "answers", str), 
        tags_fn="type", 
        groups=[
            Group(
                tasks=[
                    Task(name="ocrbench_v2"),
                    Task(name="ocrbench_v2_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="mmifeval", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "gsh/MMIfEval"), hf_split="test"),
        media_fn=("image", load_image_from_bytes),
        score_fn=("mmifeval_standard_eval", "score", "total_score"),
        groups=[
            Group(
                tasks=[
                    Task(name="mmifeval"),
                ]
            )
        ]
    ), 
    Dataset(name="cii_bench", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "m-a-p/CII-Bench"), hf_split="test"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn="accuracy",
        tags_fn=(lambda x: get_cii_bench_tags(x, "image_type") + get_cii_bench_tags(x, "rhetoric") + get_cii_bench_tags(x, "difficulty")),
        groups=[
            Group(
                tasks=[
                    Task(name="cii_bench"),
                    Task(name="cii_bench_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="blink_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/BLINK"), hf_split="val"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn=("accuracy", "score"),
        tags_fn="sub_task",
        media_fn=blink_image_fn,
        groups=[
            Group(
                tasks=[
                    Task(name="blink_val"),
                    Task(name="blink_val_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="visualsimpleqa", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "WYLing/VisualSimpleQA"), hf_split="train"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn=("accuracy", "score"),
        groups=[
            Group(
                tasks=[
                    Task(name="visualsimpleqa"),
                    Task(name="visualsimpleqa_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="realworldqa", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/RealWorldQA"), hf_split="test"),
        metric=Metric(name="exact_match", factor=100, higher_is_better=True),
        score_fn="exact_match",
        groups=[
            Group(
                tasks=[
                    Task(name="realworldqa"),
                    Task(name="realworldqa_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="refcoco_bbox_rec_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/RefCOCO_rec"), hf_split="val"),
        metric=Metric(name="refcoco_IoU", factor=100, higher_is_better=True),
        score_fn=("refcoco_IoU", "iou"),
        gt_anno_fn=("refcoco_IoU", "bbox", make_list),
        anno_fn=("refcoco_IoU", "pred", make_list),
        groups=[
            Group(
                tasks=[
                    Task(name="refcoco_bbox_rec_val"),
                    Task(name="refcoco_bbox_rec_val_mimo"),
                ]
            )
        ]
    ),
    Dataset(name="refcocog_bbox_rec_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/RefCOCOg_rec"), hf_split="val"),
        metric=Metric(name="refcoco_IoU", factor=100, higher_is_better=True),
        score_fn=("refcoco_IoU", "iou"),
        gt_anno_fn=("refcoco_IoU", "bbox", make_list),
        anno_fn=("refcoco_IoU", "pred", make_list),
        groups=[
            Group(
                tasks=[
                    Task(name="refcocog_bbox_rec_val"),
                    Task(name="refcocog_bbox_rec_val_mimo"),
                ]
            )
        ]
    ),
    Dataset(name="refcoco+_bbox_rec_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/RefCOCOplus_rec"), hf_split="val"),
        metric=Metric(name="refcoco_IoU", factor=100, higher_is_better=True),
        score_fn=("refcoco_IoU", "iou"),
        gt_anno_fn=("refcoco_IoU", "bbox", make_list),
        anno_fn=("refcoco_IoU", "pred", make_list),
        groups=[
            Group(
                tasks=[
                    Task(name="refcoco+_bbox_rec_val"),
                    Task(name="refcoco+_bbox_rec_val_mimo"),
                ]
            )
        ]
    ),
    Dataset(name="screenspot",
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "rootsautomation/ScreenSpot"), hf_split="test"),
        tags_fn=["data_type", "data_source"],
        groups=[
            Group(
                name="rec",
                support_score=True,
                metric=Metric(name="screenspot_Center_ACC", factor=100, higher_is_better=True),
                score_fn=("screenspot_Center_ACC", "iou"),
                gt_anno_fn=("screenspot_Center_ACC", "bbox", make_list),
                anno_fn=("screenspot_Center_ACC", "pred", make_list),
                tasks=[
                    Task(name="screenspot_rec_test"),
                    Task(name="screenspot_rec_test_mimo"),
                ]
            ),
            Group(
                name="reg",
                support_score=False,
                metric=Metric(name="screenspot_CIDEr", factor=100, higher_is_better=True),
                tasks=[
                    Task(name="screenspot_reg_test"),
                ]
            )
        ]
    ),
    Dataset(name="screenspot-v2",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/ScreenSpot-v2"), hf_split="test"),
        metric=Metric(name="screenspot_IoU", factor=100, higher_is_better=True),
        score_fn=("screenspot_IoU", "iou"),
        gt_anno_fn=("screenspot_IoU", "bbox", make_list),
        anno_fn=("screenspot_IoU", "pred", make_list),
        groups=[
            Group(
                name="rec",
                tasks=[
                    Task(name="screenspot_v2_rec_test"),
                    Task(name="screenspot_v2_rec_test_mimo"),
                ]
            )
        ]
    ),
    Dataset(name="screenspot-pro",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/ScreenSpot-Pro"), hf_split="test"),
        metric=Metric(name="screenspot_IoU", factor=100, higher_is_better=True),
        score_fn=("screenspot_IoU", "iou"),
        gt_anno_fn=("screenspot_IoU", "bbox", make_list),
        anno_fn=("screenspot_IoU", "pred", make_list),
        groups=[
            Group(
                name="en",
                tasks=[
                    Task(name="screenspot_pro_en_rec_test"),
                    Task(name="screenspot_pro_en_rec_test_mimo"),
                ]
            ),
            Group(
                name="cn",
                tasks=[
                    Task(name="screenspot_pro_cn_rec_test"),
                    Task(name="screenspot_pro_cn_rec_test_mimo"),
                ]
            )
        ]
    ),
    Dataset(name="vmcbench",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "suyc21/VMCBench"), hf_split="dev"),
        metric=Metric(name="average", factor=100, higher_is_better=True),
        score_fn=("average", "score"),
        groups=[
            Group(
                tasks=[
                    Task(name="vmcbench"),
                    Task(name="vmcbench_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="mathvista_testmini", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "AI4Math/MathVista"), hf_split="testmini"),
        media_fn="decoded_image",
        score_fn=("gpt_eval_score", "true_false"),
        groups=[
            Group(
                tasks=[
                    Task(name="mathvista_testmini_cot"),
                    Task(name="mathvista_testmini_format"),
                    Task(name="mathvista_testmini_solution"),
                ]
            )
        ]
    ), 
    Dataset(name="mathvision", 
        support_score=True,
        metric=Metric(name="mathvision_standard_eval", factor=100, higher_is_better=True),
        score_fn=("mathvision_standard_eval", "score"),
        media_fn="decoded_image",
        groups=[
            Group(
                name="testmini",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "MathLLMs/MathVision"), hf_split="testmini"),
                tasks=[
                    Task(name="mathvision_testmini", metric=Metric(name="mathvision_standard_eval", factor=1, higher_is_better=True), score_fn=("mathvision_standard_eval", "scores", (lambda x: x[0]))),
                    Task(name="mathvision_reason_testmini", metric=Metric(name="mathvision_gpt_eval_score", factor=1, higher_is_better=True), score_fn=("mathvision_gpt_eval_score", "scores", (lambda x: x[0]))),
                ]
            ), 
            Group(
                name="test",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "MathLLMs/MathVision"), hf_split="test"),
                tasks=[
                    Task(name="mathvision_test", metric=Metric(name="mathvision_standard_eval", factor=1, higher_is_better=True), score_fn=("mathvision_standard_eval", "scores", (lambda x: x[0]))),
                    Task(name="mathvision_reason_test", metric=Metric(name="mathvision_gpt_eval_score", factor=1, higher_is_better=True), score_fn=("mathvision_gpt_eval_score", "scores", (lambda x: x[0]))),
                ]
            ), 
        ]
    ), 
    Dataset(name="charxiv",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "tobiaslee/charxiv_qas"), hf_split="train"),
        metric=Metric(name="overall", factor=100, higher_is_better=True),
        score_fn="overall",
        tags_fn="qtype", 
        groups=[
            Group(
                tasks=[
                    Task(name="charxiv"),
                    Task(name="charxiv_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="mathverse_testmini",
        support_score=True,
        metric=Metric(name="gpt_eval_score", factor=1, higher_is_better=True),
        score_fn=("gpt_eval_score", "true_false"),
        media_fn="decoded_image",
        groups=[
            Group(
                name="default",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "CaraJ/MathVerse-lmmseval"), hf_name="testmini", hf_split="testmini"),
                tasks=[
                    Task(name="mathverse_testmini"),
                ]
            ), 
            Group(
                name="vision_only",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "CaraJ/MathVerse-lmmseval"), hf_name="testmini_version_split", hf_split="vision_only"),
                tasks=[
                    Task(name="mathverse_testmini_vision_only"),
                ]
            )
        ]
    ), 
    Dataset(name="vl_rewardbench",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "MMInstruction/VL-RewardBench"), hf_split="test"),
        metric=Metric(name="vlreward_score", factor=100, higher_is_better=True),
        score_fn=("vlreward_score", "score"),
        groups=[
            Group(
                tasks=[
                    Task(name="vl_rewardbench"),
                    Task(name="vl_rewardbench_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="cvbench", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "nyu-visionx/CV-Bench"), hf_split="test"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn="accuracy",
        tags_fn="task",
        groups=[
            Group(
                tasks=[
                    Task(name="cvbench"),
                    Task(name="cvbench_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="mmiq", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "huanqia/MM-IQ"), hf_split="test"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn="accuracy",
        tags_fn="category",
        groups=[
            Group(
                name="en",
                tasks=[
                    Task(name="mmiq_en"),
                    Task(name="mmiq_en_boxed"),
                ]
            ), 
            Group(
                name="zh",
                tasks=[
                    Task(name="mmiq_zh"),
                    Task(name="mmiq_zh_boxed"),
                ]
            )
        ]   
    ), 
    Dataset(name="pixmo_count",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "tobiaslee/pixmo_count_eval"), hf_split="train"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn="accuracy",
        groups=[
            Group(
                tasks=[
                    Task(name="pixmo_count"),
                    Task(name="pixmo_count_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="mantis", 
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "TIGER-Lab/Mantis-Eval"), hf_split="test"),
        metric=Metric(name="accuracy", factor=100, higher_is_better=True),
        score_fn="accuracy",
        tags_fn="category",
        media_fn="images",
        groups=[
            Group(
                tasks=[
                    Task(name="mantis"),
                    Task(name="mantis_boxed"),
                ]
            )
        ]
    ), 
    Dataset(name="zerobench", 
        support_score=True,
        metric=Metric(name="exact_match_accuracy", factor=100, higher_is_better=True),
        score_fn="exact_match_accuracy",
        media_fn="question_images_decoded", 
        groups=[
            Group(
                name="ori",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "jonathan-roberts1/zerobench"), hf_split="zerobench"),
                tasks=[
                    Task(name="zerobench_ori"),
                    Task(name="zerobench_ori_boxed"),
                ]
            ), 
            Group(
                name="subq",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "jonathan-roberts1/zerobench"), hf_split="zerobench_subquestions"),
                tasks=[
                    Task(name="zerobench_subq"),
                    Task(name="zerobench_subq_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="megabench",
        support_score=True,
        raw_data=RawData(type=DATASET_TYPE.NONE),
        # raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "TIGER-Lab/MEGA-Bench"), hf_split="core_single_image"),
        media_fn=task_utils.megabench.megabench_media_fn,
        metric=Metric(name="micro_mean_score", factor=100, higher_is_better=True),
        score_fn=("micro_mean_score", "scores", "query"),
        tags_fn=[
            "taxonomy_tree_path",
            # ("tags", lambda x: x.split(";")),
        ], 
        groups=[
            Group(
                name="core_single_image",
                tasks=[
                    Task(name="megabench_core_si"),
                ]
            ),
            Group(
                name="open_single_image",
                tasks=[
                    Task(name="megabench_open_si"),
                ]
            ),
            Group(
                name="core",
                tasks=[
                    Task(name="megabench_core"),
                ]
            ),
            Group(
                name="open",
                tasks=[
                    Task(name="megabench_open"),
                ]
            ),
        ]
    ),
    Dataset(name="mmvet",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/MMVet"), hf_split="test"),
        metric=Metric(name="gpt_eval_score", factor=1, higher_is_better=True),
        score_fn=("gpt_eval_score", "score"),
        tags_fn=("capability", lambda x: x.split(",")),
        groups=[
            Group(
                tasks=[
                    Task(name="mmvet"),
                    Task(name="mmvet_boxed"),
                ]
            )
        ]
    ),
    Dataset(name="hallusion_bench",
        support_score=False,
        groups=[
            Group(
                name="image",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lmms-lab/HallusionBench"), hf_split="image"),
                tasks=[
                    Task(name="hallusion_bench_image"),
                ]
            )
        ]
    ),
    Dataset(name="mm_mind2web",
        support_score=True,
        metric=Metric(name="step_accuracy", factor=100, higher_is_better=True),
        score_fn=("step_accuracy", "step_accuracy"),
        media_fn="screenshot",
        answer_fn=task_utils.mm_mind2web.mm_mind2web_answer_fn,
        groups=[
            Group(
                name="test_task",
                raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/Multimodal-Mind2Web-filtered"), hf_split="test_task"),
                tasks=[
                    Task(name="mm_mind2web_test_task"),
                ]
            )
        ]
    ),
    Dataset(name="longvideobench",
        support_score=True,
        raw_data=RawData(type=DATASET_TYPE.NONE),
        metric=Metric(name="lvb_acc", factor=100, higher_is_better=True),
        answer_fn=("lvb_acc", "answer"),
        score_fn=("lvb_acc", lambda x: int(x["answer"] == x["parsed_pred"])),
        groups=[
            Group(
                name="val_i",
                media_fn=task_utils.longvideobench.longvideobench_i_media_fn,
                tasks=[
                    Task(name="longvideobench_val_i"),
                ]
            ),
            Group(
                name="val_v",
                media_fn=task_utils.longvideobench.longvideobench_v_media_fn,
                tasks=[
                    Task(name="longvideobench_val_v"),
                ]
            ),
        ]
    ),
    Dataset(name="cc_ocr",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "lscpku/CC-OCR"), hf_split="test"),
        metric=Metric(name="anls_strip_period", factor=100, higher_is_better=True, suffix="sp"),
        media_fn=("bytes", load_image_from_bytes),
        score_fn="anls_strip_period",
        groups=[
            Group(
                tasks=[
                    Task(name="cc_ocr"),
                ]
            )
        ]
    ),
    Dataset(name="muirbench",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "MUIRBENCH/MUIRBENCH"), hf_split="test"),
        metric=Metric(name="muirbench_score_overall", factor=100, higher_is_better=True),
        media_fn="image_list",
        score_fn=("muirbench_score_overall", "score"),
        groups=[
            Group(
                tasks=[
                    Task(name="muirbench"),
                ]
            )
        ]
    ),
    Dataset(
        name="websrc_val",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "rootsautomation/websrc"), hf_split="dev"),
        media_fn=("image", load_image_from_bytes),
        metric=Metric(name="websrc_squad_f1", factor=1, higher_is_better=True),
        score_fn=("websrc_squad_f1", "f1_score"),
        groups=[
            Group(
                tasks=[
                    Task(name="websrc_val"),
                ]
            )
        ]
    ), 
    Dataset(
        name="visualwebbench_action_ground",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="action_ground", hf_split="test"),
        metric=Metric(name="accuracy", factor=1, higher_is_better=True),
        score_fn=("accuracy", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_action_ground"),
                ]
            )
        ]
    ), 
    Dataset(
        name="visualwebbench_action_prediction",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="action_prediction", hf_split="test"),
        metric=Metric(name="accuracy", factor=1, higher_is_better=True),
        score_fn=("accuracy", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_action_prediction"),
                ]
            )
        ]
    ), 
    Dataset(
        name="visualwebbench_element_ground",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="element_ground", hf_split="test"),
        metric=Metric(name="accuracy", factor=1, higher_is_better=True),
        score_fn=("accuracy", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_element_ground"),
                ]
            )
        ]
    ), 
    Dataset(
        name="visualwebbench_element_ocr",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="element_ocr", hf_split="test"),
        metric=Metric(name="rouge_l", factor=1, higher_is_better=True),
        score_fn=("rouge_l", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_element_ocr"),
                ]
            )
        ]
    ),
    Dataset(
        name="visualwebbench_heading_ocr",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="heading_ocr", hf_split="test"),
        metric=Metric(name="rouge_l", factor=1, higher_is_better=True),
        score_fn=("rouge_l", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_heading_ocr"),
                ]
            )
        ]
    ), 
    Dataset(
        name="visualwebbench_web_caption",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="web_caption", hf_split="test"),
        metric=Metric(name="rouge_l", factor=1, higher_is_better=True),
        score_fn=("rouge_l", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_web_caption"),
                ]
            )
        ]
    ), 
    Dataset(
        name="visualwebbench_webqa",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "visualwebbench/VisualWebBench"), hf_name="webqa", hf_split="test"),
        metric=Metric(name="f1", factor=1, higher_is_better=True),
        score_fn=("f1", (lambda x: x/100)),
        groups=[
            Group(
                tasks=[
                    Task(name="visualwebbench_webqa"),
                ]
            )
        ]
    ),
    Dataset(name="android_control",
        support_score=True,
        raw_data=RawData(hf_path=os.path.join(LMMS_EVAL_DATA_HOME, "custom/AndroidControl"), hf_split="test"),
        metric=Metric(name="step_wise_acc", factor=100, higher_is_better=True),
        score_fn="step_wise_acc",
        input_fn=("input", lambda x: x.split("Task:")[1]),
        answer_fn=("doc", "gt_action", lambda x: json.dumps(x)),
        media_fn=("screenshot_b64", load_image_from_bytes),
        gt_anno_fn=android_control_gt_anno_fn,
        anno_fn=android_control_anno_fn,
        tags_fn=[
            ("gt_action", "action_type")
        ],
        groups=[
            Group(
                name="default",
                tasks=[
                    Task(name="android_control"),
                ]
            ),
            Group(
                name="high",
                tasks=[
                    Task(name="android_control_high"),
                ]
            ),
            Group(
                name="low",
                tasks=[
                    Task(name="android_control_low"),
                ]
            )
        ]
    ),
]


TASK_KEYS = ["raw_data", "metric", "media_fn", "gt_anno_fn", "anno_fn", "score_fn", "tags_fn", "input_fn", "answer_fn", "output_fn", "support_score"]


def load_dataset_info():
    """
    加载所有数据集的配置信息
    """
    SUPPORT_DATASETS = []
    SUPPORT_SCORE_DATASETS = []
    METRICS = {}
    TASK_NAME_TO_TASK = {}

    _task_name_to_group = {}
    _task_name_to_dataset = {}
    for dataset in DATASETS:
        for group in dataset.groups:
            for task in group.tasks:
                SUPPORT_DATASETS.append(task.name)
                TASK_NAME_TO_TASK[task.name] = task
                _task_name_to_group[task.name] = group
                _task_name_to_dataset[task.name] = dataset
    
    for task_name, task in TASK_NAME_TO_TASK.items():
        for key in TASK_KEYS:
            if getattr(task, key) is not None:
                continue
            group = _task_name_to_group[task_name]
            if getattr(group, key) is not None:
                setattr(task, key, getattr(group, key))
                continue
            dataset = _task_name_to_dataset[task_name]
            if getattr(dataset, key) is not None:
                setattr(task, key, getattr(dataset, key))
                continue
            if getattr(_default_dataset, key) is not None:
                setattr(task, key, getattr(_default_dataset, key))
                continue
    
    for task_name, task in TASK_NAME_TO_TASK.items():
        if task.support_score:
            SUPPORT_SCORE_DATASETS.append(task_name)
            
                
    return SUPPORT_DATASETS, SUPPORT_SCORE_DATASETS, METRICS, TASK_NAME_TO_TASK


def load_dataset(task):
    """
    加载task_name对应的原始数据集，主要用于读取图片
    """
    if task.raw_data is not None:
        if task.raw_data.type == DATASET_TYPE.HF:
            return datasets.load_dataset(task.raw_data.hf_path, name=task.raw_data.hf_name)[task.raw_data.hf_split]
        else:
            return {}
    return None


def get_all_logs():
    """
    从log目录中读取所有的模型和数据集
    """
    # samples = glob.glob(os.path.join(st.session_state.log_dir, "**", "*.jsonl"), recursive=True)
    model_names = os.listdir(st.session_state.log_dir)
    model_paths = [os.path.join(st.session_state.log_dir, name) for name in model_names if os.path.isdir(os.path.join(st.session_state.log_dir, name))]
    samples = []
    for model_path in model_paths:
        for file in os.listdir(model_path):
            if file.endswith(".jsonl"):
                samples.append(os.path.join(model_path, file))

    logs = {}
    datasets_set, models_set = set(), set()
    for sample in samples:
        sample_file_basename = os.path.basename(sample)
        sample_file_dirname = os.path.dirname(sample)
        model_name = os.path.basename(sample_file_dirname)
        models_set.add(model_name)
        if model_name not in logs:
            logs[model_name] = {}
        
        sample_name_splits = os.path.splitext(sample_file_basename)[0].split('_')
        if len(sample_name_splits) < 3:
            continue
        if sample_name_splits[2] == "samples":
            exp_id = sample_name_splits[0] + '_' + sample_name_splits[1]
            dataset_name = '_'.join(sample_name_splits[3:])
        else:
            exp_id = sample_name_splits[0] + '_' + sample_name_splits[1] + '_' + sample_name_splits[2]
            dataset_name = '_'.join(sample_name_splits[4:])
        datasets_set.add(dataset_name)

        result_path = os.path.join(sample_file_dirname, f"{exp_id}_results.json")
        result = result_path if os.path.exists(result_path) else None
        
        logs[model_name][dataset_name] = (sample, result)
    
    return logs, sorted(list(datasets_set)), sorted(list(models_set))



def load_result(logs, model, dataset):
    """
    从model对应的result文件中读取dataset数据集的评测指标及得分
    """
    if model not in logs:
        return None
    if dataset not in logs[model]:
        return None
    sample_file, result_file = logs[model][dataset]
    with open(result_file, "r") as f:
        all_results = json.load(f)
    sample_count_original = all_results["n-samples"][dataset]["original"]
    sample_count_effective = all_results["n-samples"][dataset]["effective"]
    higher_is_better = all_results["higher_is_better"][dataset]
    metrics = [(k, v) for k, v in higher_is_better.items() if not k.startswith("bypass")]

    results, _metrics = [], []
    # print(all_results["results"][dataset])
    for metric in metrics:
        if f"{metric[0]},none" in all_results["results"][dataset]:
            results.append(all_results["results"][dataset][f"{metric[0]},none"])
            _metrics.append(metric)
        elif f"{metric[0]},flexible-extract" in all_results["results"][dataset]:
            results.append(all_results["results"][dataset][f"{metric[0]},flexible-extract"])
            _metrics.append(metric)
        else:
            pass
            # raise ValueError(f"Metric {metric[0]} not found in results")
    _results = {metric: result for metric, result in zip(_metrics, results)}
    return sample_count_original, sample_count_effective, _results



def process_images(sample, task, cached_dataset):
    if cached_dataset is None:
        return None
    if isinstance(cached_dataset, datasets.Dataset):
        item = cached_dataset[sample["doc_id"]]
        return get_media(task.media_fn, item)
    else:
        return task.media_fn(sample, cached_dataset)



def process_sample(sample, task, cached_dataset):
    """
    处理样本，每一条样本输出包括
    - images: 图片列表
    - input: 输入
    - answer: 答案
    - response: 模型输出
    - score: 样本得分
    """
    sample = {
        "input": apply(sample, task.input_fn),
        "answer": apply(sample, task.answer_fn),
        "response": apply(sample, task.output_fn),
        "score": apply(sample, task.score_fn) if task.score_fn is not None else None,
        "tags": get_tags(task.tags_fn, sample) if task.tags_fn is not None else None,
        "gt_annotations": apply(sample, task.gt_anno_fn) if task.gt_anno_fn is not None else None,
        "annotations": apply(sample, task.anno_fn) if task.anno_fn is not None else None,
    }
    return sample



def load_samples(logs, model, dataset):
    """
    从model和dataset对应的samples文件中读取模型输出
    """
    if model not in logs:
        return None
    if dataset not in logs[model]:
        return None
    sample_file, result_file = logs[model][dataset]
    with open(sample_file, "r") as f:
        samples = [json.loads(line) for line in f]
    samples = {sample["doc_id"]: sample for sample in samples}
    return samples