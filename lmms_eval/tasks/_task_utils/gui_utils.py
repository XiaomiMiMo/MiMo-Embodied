from typing import List, Literal
from PIL import Image
import json
import re
import math

SYSTEM_PROMPT = "You are a helpful assistant."

prompt_uitars = """You are a GUI agent. You are given a task and your action history, with screenshots. 
You need to perform the next action to complete the task. 

## Output Format

Thought: ...
Action: ...


## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(direction='down or up or right or left')
press_back()
press_home()
wait()
finished() # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in Thought part.

- Summarize your next action (with its target element) in one sentence in Thought part.

## User Instruction
{goal}"""

prompt_mimo_with_instruction = """You are a GUI agent. You will be provided with a screenshot, a goal, your action history, and an instruction for your next action. You need to perform the next action to complete the task.

## Action Space
{action_space}

## Goal
{goal}

## Previous Actions
{previous_actions}

## Instruction for the next step
{instruction}

Now, output the next action in json format [{{\"action\": \"{{action_name}}\"}}, ...]."""

prompt_mimo_wo_instruction = """You are a GUI agent. You will be provided with a screenshot, a goal, and your action history. You need to perform the next action to complete the task.

## Action Space
{action_space}

## Goal
{goal}

## Previous Actions
{previous_actions}

Now, output the next action in json format [{{\"action\": \"{{action_name}}\"}}, ...]."""

class MiMo_Unified_GUIAction:
    # - 点击类：click(start_point=<|point_start|>(x,y)<|point_end|>)
    # - 滑动类：scroll(start_point=<|point_start|>(x,y)<|point_end|>, direction=direction)
    # - 拖拽类：drag(start_point=<|point_start|>(x,y)<|point_end|>, end_point=<|point_start|>(x,y)<|point_end|>)
    # - 输入类：input(start_point=<|point_start|>(x,y)<|point_end|>, text=text)
    # - 按键类：press(keys=[key1, key2, ...]) # 既包含ctrl+c这种组合键，也包含home、enter等
    # - 长按类：longpress(start_point=<|point_start|>(x,y)<|point_end|>)
    # - 结束类：finished(status="xxxx")
    # - 等待类：wait() #用于处理广告等需要等一会儿的
    # - 打开app类：open(app="xxxx")
    # - 切换app类：appswitch()

    action_name: Literal["click", "scroll", "drag", "input", "press", "longpress", "finished", "wait", "open", "appswitch", "hover", "select"]
    start_point: tuple[float, float] | None = None
    end_point: tuple[float, float] | None = None
    direction: Literal["up", "down", "left", "right"] | None = None
    scroll_distance: float | None = None
    text: str | None = None
    keys: list[str] | None = None
    status: str | None = None
    app: str | None = None

def mimo_2_uitars(mimo_action: dict):
    """
    Convert MiMo action to UITARS action format
    """
    action = mimo_action.get("action", None)
    action_type = action.get('action_name', '')
    
    if action_type == 'click':
        start_point = action.get('start_point', [0, 0])
        x, y = start_point[0], start_point[1]
        click_x = int(x* 1000)
        click_y = int(y* 1000)
        return f"click(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
    
    elif action_type == 'input':
        text = action.get('text', '')
        return f"type(content='{text}')"
    
    elif action_type == 'scroll':
        direction = action.get('direction', 'down')
        return f"scroll(direction={direction})"
    
    elif action_type == 'longpress':
        start_point = action.get('start_point', [0, 0])
        x, y = start_point[0], start_point[1]
        click_x = int(x* 1000)
        click_y = int(y* 1000)
        return f"long_press(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
    
    elif action_type == 'press':
        keys = action.get('keys', [])
        if keys == ['home']:
            return f"press_home()"
        elif keys == ['back']:
            return f"press_back()"
        elif keys == ['enter']:
            return f"press_enter()"
    
    elif action_type == 'open':
        app_name = action.get('app_name', '')
        return f"open(app_name='{app_name}')"
    
    elif action_type == 'wait':
        return "wait()"
    
    elif action_type == 'finished':
        return "finished()"
    
    else:
        print('action:', action)
        raise NotImplementedError(f"Unknown action type: {action_type}")

def uitars2mimo(action_str):
    """
    将ui-tars action字符串转换为mimo schema格式的action字典
    """
    action = {}

    def extract_coords(s):
        action_str = s.split("Action:")[-1].strip()
        first_bracket = action_str.find("(")
        start = action_str.find("(", first_bracket + 1)
        end = action_str.find(")")
        if start != -1 and end != -1:
            coords_str = action_str[start+1:end].strip()
            x, y = coords_str.split(",")
            # 归一化
            return [float(x)/1000, float(y)/1000]
        raise ValueError(f"Cannot find coordinates in the string: {s}")

    if "click(" in action_str:
        action["action_name"] = "click"
        action["start_point"] = extract_coords(action_str)
    elif "long_press(" in action_str:
        action["action_name"] = "longpress"
        action["start_point"] = extract_coords(action_str)
    elif "type(" in action_str:
        action["action_name"] = "input"
        text = action_str.split("content=\'")[1].split("\'")[0]
        action["text"] = text
    elif "scroll(" in action_str:
        action["action_name"] = "scroll"
        if "direction=\'" in action_str:
            direction = action_str.split("direction=\'")[1].split("\'")[0]
            action["direction"] = direction
        else:
            direction = action_str.split("direction=")[1].split(")")[0]
            action["direction"] = direction
    elif "press_back()" in action_str:
        action["action_name"] = "press"
        action["keys"] = ["back"]
    elif "press_home()" in action_str:
        action["action_name"] = "press"
        action["keys"] = ["home"]
    elif "press_enter()" in action_str:
        action["action_name"] = "press"
        action["keys"] = ["enter"]
    elif "wait()" in action_str:
        action["action_name"] = "wait"
    elif "finished()" in action_str:
        action["action_name"] = "finished"
    elif "open(app_name=" in action_str:
        action["action_name"] = "open"
        app_name = action_str.split("app_name=\'")[1].split("\'")[0]
        action["app_name"] = app_name
    elif "open_app(app_name=" in action_str:
        action["action_name"] = "open"
        app_name = action_str.split("app_name=\'")[1].split("\'")[0]
        action["app_name"] = app_name
    else:
        raise NotImplementedError(f"Unknown action string: {action_str}")
    return {"action": action}

def _get_direction(point1, point2):
    try:
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        assert x1 is not None
        assert x2 is not None
        assert y1 is not None
        assert y2 is not None

        vector = (x2 - x1, y2 - y1)
        vx, vy = vector
    except Exception as e:
        return "no direction"

    directions = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (1, 0),
        "right": (-1, 0)
    }

    vector_length = math.sqrt(vx ** 2 + vy ** 2)
    if vector_length == 0:
        return "no direction"
    unit_vector = (vx / vector_length, vy / vector_length)

    max_cosine = -float('inf')
    closest_direction = None
    for direction, dir_vector in directions.items():
        dx, dy = dir_vector
        dir_length = math.sqrt(dx ** 2 + dy ** 2)
        cos_theta = (unit_vector[0] * dx + unit_vector[1] * dy) / dir_length
        if cos_theta > max_cosine:
            max_cosine = cos_theta
            closest_direction = direction

    return closest_direction

def mimo_agent2mimo(mimo_action: dict, image_width: int, image_height: int):
    """
    Convert MiMoAgent action to MiMo action format
    """
    # json_pattern = re.compile(r'```json(.*)```')
    # json_str = json_pattern.search(mimo_action, re.DOTALL).group(1).strip()
    if "```json" in mimo_action:
        json_str = re.search(r'```json(.*)```', mimo_action, re.DOTALL).group(1).strip()
    else:
        start_idx = mimo_action.find("{\"action\":")
        end_idx = mimo_action.rfind("}")
        json_str = mimo_action[start_idx:end_idx+1]
    think_content = re.search(r'<think>(.*)</think>', mimo_action, re.DOTALL)
    if think_content:
        think_content = think_content.group(1).strip()
    else:
        think_content = ""
    json_dict = json.loads(json_str)
    if isinstance(json_dict, list):
        json_dict = json_dict[0]
    json_dict["action_name"] = json_dict.pop("action")
    json_dict["think"] = think_content
    if "start_point" in json_dict:
        json_dict["start_point"] = [json_dict["start_point"][0]/image_width, json_dict["start_point"][1]/image_height]
    if "end_point" in json_dict:
        json_dict["end_point"] = [json_dict["end_point"][0]/image_width, json_dict["end_point"][1]/image_height]
    if "app" in json_dict:
        json_dict["app_name"] = json_dict.pop("app")
    if "direction" in json_dict:
        if json_dict["direction"] in ["up", "down"]:
            json_dict["direction"] = "down" if json_dict["direction"] == "up" else "up"
    if json_dict["action_name"] == "drag":
        json_dict["action_name"] = "scroll"
        start_point = json_dict["start_point"]
        end_point = json_dict["end_point"]
        direction = _get_direction(start_point, end_point)
        json_dict["direction"] = direction
    return json_dict
    
    

def build_history_actions_str(history_list):
    history = []
    
    # Get indices of the last 4 image records
    image_indices = range(max(0, len(history_list) - 4), len(history_list))
    
    for i, step_history in enumerate(history_list):
     # If current index is in the last 4 image records, add the image
        if i in image_indices:
            image_path = step_history["image_path"]
            image = Image.open(image_path)
            image_history = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    }
                ]
            }
            history.append(image_history)
        
        # Add action
        if i in image_indices:
            action = mimo_2_uitars(step_history)
            thought = step_history.get("instruction", "")
            text_history = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thought: {thought}\nAction: {action}"}
                ]
            }
            history.append(text_history)
    
    return history

def transfer_action_into_absolute_style(action: dict, width:int, height:int, convert_scroll: bool = False):
    action_without_none = {k: v for k, v in action.items() if v is not None}
    action = action_without_none
    for point in ["start_point", "end_point"]:
        if point in action:
            try:
                relative_w, relative_h = action[point]
                absolute_w, absolute_h = int(round(relative_w * width)), int(round(relative_h * height))
                action[point] = [absolute_w, absolute_h]
            except Exception as e:
                print(f"Error transferring action into absolute style: {e}")
                print(f"Action: {action}")
                raise e
    if action["action_name"] == "input" and "start_point" in action:
        action.pop("start_point")
    
    if convert_scroll and action["action_name"] == "scroll":
        if "start_point" not in action and "end_point" not in action:
            action["start_point"] = [int(round(0.5*width)), int(round(0.5*height))]
            if action["direction"] == "up":
                action["end_point"] = [int(round(0.5*width)), int(round(0.8*height))]
            elif action["direction"] == "down":
                action["end_point"] = [int(round(0.5*width)), int(round(0.2*height))]
            elif action["direction"] == "left":
                action["end_point"] = [int(round(0.8*width)), int(round(0.5*height))]
            elif action["direction"] == "right":
                action["end_point"] = [int(round(0.2*width)), int(round(0.5*height))]
        action["action_name"] = "drag"
        if "scroll_distance" in action:
            action.pop("scroll_distance")
        if "direction" in action:
            action.pop("direction")
        action["end_point"] = action.pop("end_point")
    
    if not convert_scroll and action["action_name"] == "scroll":
        if "scroll_distance" in action:
            if action["direction"] in ["right", "left"]:
                action["scroll_distance"] = int(round(action["scroll_distance"] * width))
            elif action["direction"] in ["down", "up"]:
                action["scroll_distance"] = int(round(action["scroll_distance"] * height))
        if "direction" in action:
            if action["direction"] in ["right", "left"]:
                action["direction"] = "right" if action["direction"] == "right" else "left"
            elif action["direction"] in ["down", "up"]:
                action["direction"] = "down" if action["direction"] == "up" else "down"
        if "scroll_distance" in action:
            action.pop("scroll_distance")
    return action

def transfer_history_data_into_absolute_style(history_action_list: List[str], width:int, height:int, convert_scroll: bool = False):
    result_history_action_list = []
    for history_action in history_action_list:
        # 首先，匹配从 {"action" 开头到其后第一个 }的字段，并用json进行load
        structured_action = transfer_action_into_absolute_style(history_action, width, height, convert_scroll)
        if "action_explanation" in structured_action:
            structured_action.pop("action_explanation")
        json_action = json.dumps(structured_action, ensure_ascii=False)
        if 'action_explanation' in history_action:
            json_action = f'{history_action["action_explanation"]} {json_action}'
        result_history_action_list.append(json_action)

    return result_history_action_list
    

class ActionEvaluator:
    @staticmethod
    def compute_atomic_metrics(step_results):
        pass