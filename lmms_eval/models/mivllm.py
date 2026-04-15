# Copyright 2025 Xiaomi Corporation.


from typing import List, Optional, Tuple, Literal

import numpy as np
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import torch
from transformers import AutoProcessor

from lmms_eval.models.model_utils.qwen.vision_process import process_vision_info

try:
    from vllm import LLM, SamplingParams
    import warnings
    warnings.filterwarnings("ignore", message=".*The following intended overrides are not keyword-only args.*")

    from vllm.entrypoints.chat_utils import resolve_chat_template_content_format
    from vllm.inputs.data import TextPrompt, TokensPrompt
except ImportError:
    vllm = None




from torch.utils.data import Dataset, DataLoader

def transform_video(video):
    # transform TCHW torch tensor ([0,255]) to THWC numpy uint8 array
    video = video.cpu().numpy()
    video = video.transpose(0, 2, 3, 1)
    video = video.astype(np.uint8)
    return video


class MiVLLMDataset(Dataset):
    def __init__(self, requests, model):
        self.requests = requests
        self.model = model
        self.processor = model.processor
        self.thinking_prompt = model.thinking_prompt
        self.thinking_prompt_user = model.thinking_prompt_user
        self.media_position = model.media_position
        self.image_mm_processor_kwargs = model.image_mm_processor_kwargs
        self.video_mm_processor_kwargs = model.video_mm_processor_kwargs
    
    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = self.requests[idx].arguments
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 32768
        if gen_kwargs["max_new_tokens"] > 16384:
            gen_kwargs["max_new_tokens"] = 16384
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = 1.0

        params = {
            "temperature": gen_kwargs["temperature"],
            "max_tokens": gen_kwargs["max_new_tokens"],
            "top_p": gen_kwargs["top_p"],
        }
        sampling_params = SamplingParams(**params)
        
        content = []

        visuals = [doc_to_visual(self.model.task_dict[task][split][doc_id])]
        if None not in visuals:
            visuals = self.model.flatten(visuals)
            for visual in visuals:
                if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".flv", ".wmv")):
                    content.append({"type": "video", "video": visual, **self.video_mm_processor_kwargs})
                elif isinstance(visual, Image.Image):
                    content.append({"type": "image", "image": visual, **self.image_mm_processor_kwargs})
        content.append({"type": "text", "text": contexts})

        text_content = [_ for _ in content if _.get("type") == "text"]
        media_content = [_ for _ in content if _.get("type") != "text"]
        if self.media_position in ["first", "last"]:
            content = media_content + text_content if self.media_position == "first" else text_content + media_content
        elif self.media_position == "interleaved":
            # TODO: only work for single text content and when the number of media content == the number of media tokens
            interleaved_content = []
            media_idx = 0
            for _text_content in text_content:
                text = _text_content["text"]
                for j in range(32):
                    text = text.replace(f"<image {j}>", f"<image>").replace(f"\\<image {j}\\>", "<image>").replace("<|image|>", "<image>")
                split_text = text.split("<image>")
                for i in range(len(split_text)):
                    interleaved_content.append({"type": "text", "text": split_text[i]})
                    if media_idx < len(media_content):
                        interleaved_content.append(media_content[media_idx])
                    media_idx += 1
            if media_idx > len(media_content) + 1:
                eval_logger.warning(f"Number of media content is less than the number of media tokens. media_idx: {media_idx}, len(media_content): {len(media_content)}; doc_id: {doc_id}, task: {task}, split: {split}")
                # print(f"text_content: {text_content}\nmedia_content: {media_content}\ninterleaved_content: {interleaved_content}\nmedia_idx: {media_idx}")  # for debug
                content = interleaved_content
            elif media_idx < len(media_content):
                content = media_content[media_idx:] + interleaved_content
            else:
                content = interleaved_content
        else:
            raise ValueError(f"Invalid media position: {self.media_position}")
        
        content.append({"type": "text", "text": self.thinking_prompt_user})
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        _prompt = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        _prompt = _prompt + self.thinking_prompt
            
        prompt = TextPrompt(prompt=_prompt)

        images, videos, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        video_kwargs["fps"] = self.video_mm_processor_kwargs["fps"]
        mm_data = {}
        if images is not None and len(images) > 0:
            mm_data["image"] = images
        if videos is not None and len(videos) > 0:
            mm_data["video"] = [transform_video(video) for video in videos]

        if len(mm_data) > 0:
            prompt["multi_modal_data"] = mm_data
            prompt["mm_processor_kwargs"] = video_kwargs
        
        return idx, message, prompt, sampling_params


def mivllm_collate_fn(batch):
    idxs = [item[0] for item in batch]
    messages = [item[1] for item in batch]
    prompts = [item[2] for item in batch]
    sampling_params = batch[0][3]
    return idxs, messages, prompts, sampling_params


@register_model("mivllm")
class MiVLLM(lmms):
    def __init__(
        self,
        model_version: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        dtype: Optional[str] = "auto",
        max_images: int = 32,
        max_videos: int = 8,
        max_audios: int = 8,
        max_model_len: int = 32768,
        max_num_seqs: int = 8,

        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        video_total_max_pixels: Optional[int] = None,
        video_fps: Optional[int] = None,
        video_min_frames: Optional[int] = None,
        video_max_frames: Optional[int] = None,
        video_nframes: Optional[int] = None,
        enforce_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        disable_thinking: bool = False,
        enforce_thinking_user: bool = False,
        disable_thinking_user: bool = False,
        media_position: Literal["first", "last", "interleaved"] = "first",

        num_workers: int = 8,
        prefetch_factor: int = 4,
        
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.max_images = max_images
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        
        self.image_mm_processor_kwargs = self._get_image_mm_processor_kwargs(image_min_pixels, image_max_pixels)
        self.video_mm_processor_kwargs = self._get_video_mm_processor_kwargs(video_min_pixels, video_max_pixels, video_total_max_pixels, video_fps, video_min_frames, video_max_frames, video_nframes)
        self.media_position = media_position
        self.enforce_thinking = enforce_thinking
        self.thinking_budget = thinking_budget
        self.disable_thinking = disable_thinking
        self.enforce_thinking_user = enforce_thinking_user
        self.disable_thinking_user = disable_thinking_user
        
        if self.enforce_thinking:
            if self.thinking_budget is not None:
                self.thinking_prompt = f"<think_{self.thinking_budget}>"
            else:
                self.thinking_prompt = "<think>"
        elif self.disable_thinking:
            self.thinking_prompt = "<think>\n</think>\n"
        else:
            self.thinking_prompt = ""
        
        if self.enforce_thinking_user:
            self.thinking_prompt_user = " /think"
        elif self.disable_thinking_user:
            self.thinking_prompt_user = " /no_think"
        else:
            self.thinking_prompt_user = ""
        
        # FIXME
        mm_processor_kwargs = self.image_mm_processor_kwargs

        accelerator = Accelerator()
        self.client = LLM(
            model=self.model_version,
            device=accelerator.device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_images, "video": max_videos, "audio": max_audios},
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            enforce_eager=True,
            max_num_seqs=max_num_seqs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_version)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            # self._rank = self.accelerator.local_process_index
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
    
    def _get_image_mm_processor_kwargs(self, image_min_pixels, image_max_pixels):
        image_mm_processor_kwargs = {}
        if image_min_pixels is not None:
            image_mm_processor_kwargs["min_pixels"] = image_min_pixels
        if image_max_pixels is not None:
            image_mm_processor_kwargs["max_pixels"] = image_max_pixels
        return image_mm_processor_kwargs
    
    def _get_video_mm_processor_kwargs(self, video_min_pixels, video_max_pixels, video_total_max_pixels, video_fps, video_min_frames, video_max_frames, video_nframes):
        video_mm_processor_kwargs = {}
        if video_min_pixels is not None:
            video_mm_processor_kwargs["min_pixels"] = video_min_pixels
        if video_max_pixels is not None:
            video_mm_processor_kwargs["max_pixels"] = video_max_pixels
        if video_total_max_pixels is not None:
            video_mm_processor_kwargs["total_pixels"] = video_total_max_pixels
        if video_fps is not None:
            print(f"video_fps: {video_fps}")
            video_mm_processor_kwargs["fps"] = video_fps
            if video_min_frames is not None:
                video_mm_processor_kwargs["min_frames"] = video_min_frames
            if video_max_frames is not None:
                video_mm_processor_kwargs["max_frames"] = video_max_frames
        elif video_nframes is not None:
            print(f"video_nframes: {video_nframes}")
            video_mm_processor_kwargs["nframes"] = video_nframes
        return video_mm_processor_kwargs


    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        batch_size = self.batch_size_per_gpu

        pt_dataset = MiVLLMDataset(requests, self)
        pt_dataloader = DataLoader(
            pt_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=mivllm_collate_fn
        )
        
        for batch_requests in pt_dataloader:
            idxs, messages, prompts, sampling_params = batch_requests

            response = self.client.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            response_text = [o.outputs[0].text for o in response]
            response_text = [self.thinking_prompt + _ for _ in response_text]
            
            assert len(response_text) == len(messages)

            for idx, text in zip(idxs, response_text):
                res[idx] = text
            pbar.update(len(messages))
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Not implemented"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
