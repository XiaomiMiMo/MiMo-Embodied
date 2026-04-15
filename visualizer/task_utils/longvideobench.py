import streamlit as st
import os
import io
import decord
import torch
from decord import VideoReader, cpu
from PIL import Image


def load_video(video_file, duration, max_num_frames=16):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]

    return [Image.fromarray(fr).convert("RGB") for fr in frames]

def load_frames(video_path, duration, max_num_frames):
    video_path = os.path.join(st.session_state.lmms_eval_cache_dir, "longvideobench", "videos", video_path)
    return load_video(video_path, duration, max_num_frames)


def longvideobench_i_media_fn(sample, cached_dataset): 
    video_path = sample["doc"]["video_path"]
    duration = sample["doc"]["duration"]
    max_num_frames = sample["input"].count("<image>")
    if (video_path, max_num_frames) in cached_dataset:
        return cached_dataset[(video_path, max_num_frames)]
    frames = load_frames(video_path, duration, max_num_frames)
    cached_dataset[(video_path, max_num_frames)] = frames
    return frames


def longvideobench_v_media_fn(sample, cached_dataset): 
    video_path = sample["doc"]["video_path"]
    video_path = os.path.join(st.session_state.lmms_eval_cache_dir, "longvideobench", "videos", video_path)
    if video_path in cached_dataset:
        return [io.BytesIO(cached_dataset[video_path])]
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    cached_dataset[video_path] = video_bytes
    return [io.BytesIO(video_bytes)]