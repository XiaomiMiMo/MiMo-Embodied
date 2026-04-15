import json


def mm_mind2web_answer_fn(sample):
    target = sample['doc']["operation"]['value'].strip()
    if target == "":
        target = "None"
    bboxes = []
    width = sample['doc']['width']
    height = sample['doc']['height']
    for x, y, w, h in sample['doc']["bboxes"]:
        bboxes.append([round(x / width, 2), round(y / height, 2), round((x + w) / width, 2), round((y + h) / height, 2)])
    answer = json.dumps({
        "bboxes": bboxes,
        "action": sample['doc']["operation"]['op'],
        "value": target
    })
    return answer
