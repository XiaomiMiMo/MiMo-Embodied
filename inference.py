# Copyright 2025 Xiaomi Corporation.


import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "MiMo-XFM HF ckpt path"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

# default processor
processor = AutoProcessor.from_pretrained(model_path)

# thinking mode
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/demo.jpg",
            },
            {"type": "text", "text": "which book is cloest to the camera?"},
        ],
    }
]

# no think mode
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "assets/demo.jpg",
#             },
#             {"type": "text", "text": "which book is cloest to the camera? /no_think"},
#         ],
#     }
# ]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])