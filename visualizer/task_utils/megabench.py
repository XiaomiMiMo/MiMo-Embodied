import ast
import os
import streamlit as st
from task_utils.media_utils import load_media_from_path


def megabench_media_fn(sample, cached_dataset):
    _medias = []
    for key in ["global_media", "example_media", "query_media"]:
        _media = sample['doc'].get(key, "[]")
        if _media.startswith("[") and _media.endswith("]"):
            _media = ast.literal_eval(_media)
        if isinstance(_media, list):
            _medias.extend(_media)
        else:
            _medias.append(_media)
    medias = []
    for _media in _medias:
        if _media in cached_dataset:
            medias.append(cached_dataset[_media])
        else:
            _media_path = os.path.join(st.session_state.lmms_eval_cache_dir, "megabench_data", _media)
            media = load_media_from_path(_media_path)
            cached_dataset[_media] = media
            medias.append(media)
    return medias