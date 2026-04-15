import streamlit as st

from data_utils import get_all_logs

def init_st_session_state():
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = set()
    if "selected_datasets" not in st.session_state:
        st.session_state.selected_datasets = set()
    if "cached_dataset" not in st.session_state:
        st.session_state.cached_dataset = {}
    if "log_dir" not in st.session_state:
        st.session_state.log_dir = None
    if "lmms_eval_cache_dir" not in st.session_state:
        st.session_state.lmms_eval_cache_dir = os.getenv("HF_HOME")
        assert st.session_state.lmms_eval_cache_dir is not None, "Environment variable HF_HOME is not set"
    if "cache" not in st.session_state:
        st.session_state.cache = None


@st.dialog(title="设置评测结果路径", width="large")
def set_log_dir():
    with st.form("set_log_dir"):
        log_dir = st.text_input("评测结果路径", value=st.session_state.log_dir)
        if st.form_submit_button("确定"):
            st.session_state.log_dir = log_dir
            st.session_state.cache = get_all_logs()
            st.rerun()


@st.dialog(title="Set Results Dir", width="large")
def set_log_dir_en():
    with st.form("set_log_dir"):
        log_dir = st.text_input("Results Dir", value=st.session_state.log_dir)
        if st.form_submit_button("Confirm"):
            st.session_state.log_dir = log_dir
            st.session_state.cache = get_all_logs()
            st.rerun()


def select_models(models):
    """
    点击全选触发，更新selected_models
    """
    for model in models:
        if model not in st.session_state.selected_models:
            st.session_state.selected_models.add(model)
            st.session_state[f"model-checkbox-{model}"] = True
    st.rerun()

def deselect_models(models):
    """
    点击取消全选触发，更新selected_models
    """
    for model in models:
        if model in st.session_state.selected_models:
            st.session_state.selected_models.remove(model)
            st.session_state[f"model-checkbox-{model}"] = False
    st.rerun()

def clear_models():
    """
    点击清除模型触发，更新selected_models
    """
    st.session_state.selected_models.clear()
    for model in st.session_state:
        if model.startswith("model-checkbox-"):
            st.session_state[model] = False
    st.rerun()

def select_datasets(datasets):
    """
    点击全选触发，更新selected_datasets
    """
    for dataset in datasets:
        if dataset not in st.session_state.selected_datasets:
            st.session_state.selected_datasets.add(dataset)
            st.session_state[f"dataset-checkbox-{dataset}"] = True
    st.rerun()

def deselect_datasets(datasets):
    """
    点击取消全选触发，更新selected_datasets
    """
    for dataset in datasets:
        if dataset in st.session_state.selected_datasets:
            st.session_state.selected_datasets.remove(dataset)
            st.session_state[f"dataset-checkbox-{dataset}"] = False
    st.rerun()

def clear_datasets():
    """
    点击清除数据集触发，更新selected_datasets
    """
    st.session_state.selected_datasets.clear()
    for dataset in st.session_state:
        if dataset.startswith("dataset-checkbox-"):
            st.session_state[dataset] = False
    st.rerun()

def refresh_cache():
    """
    点击刷新缓存触发，更新cache
    """
    st.session_state.cache = get_all_logs()


# Add these functions at the top of your show_samples function (around line 151)
def string_to_color(tag_string):
    """Generate a consistent color from a string using hashing"""
    import hashlib
    # Get a hash of the string
    hash_value = int(hashlib.md5(tag_string.encode()).hexdigest(), 16)
    # Convert to HSL for better color distribution
    h = hash_value % 360  # Hue: 0-359 degrees
    s = 65 + (hash_value % 20)  # Saturation: 65-85%
    l = 45 + (hash_value % 15)  # Lightness: 45-60%
    
    # Convert HSL to RGB
    h /= 360
    s, l = s / 100, l / 100
    
    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    # Convert to hex
    r, g, b = [int(x * 255) for x in (r, g, b)]
    return f"#{r:02x}{g:02x}{b:02x}"

def is_dark_color(hex_color):
    """Determine if a color is dark (should use white text) or light (should use black text)"""
    # Remove the # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    
    # Calculate brightness (using standard formula)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Return True if color is dark
    return brightness < 0.5


import os
from PIL import Image, ImageDraw, ImageFont
def get_font():
    font_path = os.path.join(os.path.dirname(__file__), "SimHei.ttf")
    font_size = 16
    font = ImageFont.truetype(font_path, font_size)
    return font

font = get_font()
font_size = 32
def draw_bbox(draw: ImageDraw.Draw, size, bbox, label, color, thickness=5): 
    width, height = size
    if all(0 <= _ and _ <= 3 for _ in bbox):
        bbox = [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)]
    draw.rectangle(bbox, outline=color, width=thickness)
    draw.text((bbox[0], bbox[1]), label, fill=color, font=font, font_size=font_size)


def draw_point(draw: ImageDraw.Draw, size, point, label, color, radius=5):
    width, height = size
    if all(0 <= _ and _ <= 3 for _ in point):
        point = [int(point[0] * width), int(point[1] * height)]
    draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), outline=color, fill=color)
    draw.text((point[0], point[1]), label, fill=color, font=font, font_size=font_size)


def draw_circle(draw: ImageDraw.Draw, size, center, radius, label, color, thickness=5):
    width, height = size
    if all(0 <= _ and _ <= 3 for _ in center):
        center = [int(center[0] * width), int(center[1] * height)]
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline=color, width=thickness)
    draw.text((center[0] - radius, center[1] - radius), label, fill=color, font=font, font_size=font_size)

def parse_annotation(anno):
    if isinstance(anno, (list, tuple)) and len(anno) == 4:
        return "bbox", anno
    elif isinstance(anno, (list, tuple)) and len(anno) == 2:
        return "point", anno
    elif isinstance(anno, dict):
        return anno.get("type"), anno.get("value")

def draw_annotations(image: Image.Image, annotations: list):
    draw = ImageDraw.Draw(image)
    for anno, label, color in annotations:
        print(anno)
        anno_type, anno_value = parse_annotation(anno)
        if anno_type == "bbox":
            draw_bbox(draw, image.size, anno_value, label, color)
        elif anno_type == "point":
            draw_point(draw, image.size, anno_value, label, color)
        elif anno_type == "circle":
            center, radius = anno_value["center"], anno_value["radius"]
            draw_circle(draw, image.size, center, radius, label, color)
        elif anno_type == "":
            pass
    return image
