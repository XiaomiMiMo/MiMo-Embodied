import streamlit as st
from PIL import Image
import io
import base64

def load_media_from_path(media_path):
    if media_path.endswith((".jpg", ".png", ".jpeg")):
        return Image.open(media_path)
    elif media_path.endswith(".mp4"):
        with open(media_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes
    else:
        raise NotImplementedError(f"Unsupported media type: {media_path}")
    

def load_image_from_bytes(_bytes):
    with io.BytesIO(base64.b64decode(_bytes)) as f:
        img = Image.open(f)
        img.load()
        return img