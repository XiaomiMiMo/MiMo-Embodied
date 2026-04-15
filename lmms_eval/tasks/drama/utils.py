# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from pathlib import Path

# Third-party
import yaml
from PIL import Image

# Local
from lmms_eval.tasks._task_utils.eval_utils import normalize_bbox, parse_bbox

with open(Path(__file__).parent / "drama.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config["img_root"]

DRAMA_METRICS = ["ACC@0.5"]

PROMPT = "Identify the critical object in the scene. Output a JSON in the format [{{\"bbox_2d\": [...], \"caption\": \"{{caption_and_explanation}}\"}}, ...]."


def drama_doc_to_visual(doc):
    img_path = vision_cache_dir + doc["image"].split("combined")[-1]
    image = Image.open(img_path)
    return [image.convert("RGB")]


def drama_doc_to_text(doc):
    return PROMPT


def drama_process_results(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        result: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    img_path = vision_cache_dir + doc["image"].split("combined")[-1]
    image = Image.open(img_path)
    width, height = image.size
    doc["image_width"] = width
    doc["image_height"] = height
    doc["bbox"][2] = doc["bbox"][0] + doc["bbox"][2]
    doc["bbox"][3] = doc["bbox"][1] + doc["bbox"][3]

    pred = result[0] if len(result) > 0 else ""
    pred = parse_bbox(pred)
    pred = normalize_bbox(pred, doc["image_width"], doc["image_height"], resize_max_pixels=int(os.getenv("QWEN_RESIZE_MAX_PIXELS", 0)))
    bbox = normalize_bbox(doc["bbox"], doc["image_width"], doc["image_height"])
    iou = compute_iou(bbox, pred)
    data_dict = {"pred": pred, "bbox": bbox, "iou": iou}
    return {f"drama_{metric}": data_dict for metric in DRAMA_METRICS}


def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def compute_accuracy(iou, threshold=0.5):
    """
    Compute the accuracy of two bounding boxes based on a specified threshold.

    Parameters:
    - iou (float): IoU of the two bounding boxes.
    - threshold (float): Threshold for the IoU to consider the prediction correct.

    Returns:
    - float: Accuracy of the prediction based on the IoU threshold.
    """
    return iou >= threshold


def drama_aggregate_results(results):
    """
    Aggregate the results of the DRAMA evaluation task.

    Args:
    - results (list of dict): List of result dictionaries.

    Returns:
    - float: Aggregated accuracy score.
    """
    scores = 0
    for result in results:
        iou = result["iou"]
        scores += compute_accuracy(iou)
    return scores / len(results)