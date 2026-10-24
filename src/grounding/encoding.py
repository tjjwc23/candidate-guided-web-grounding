"""grounding / encoding."""

from __future__ import annotations

import torch
from PIL import Image

from grounding.coordinate_prompt import get_computer_use_system_prompt
from manifest import ROOT as REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT


DISPLAY_SIZE = 1000
A_PREFIX = (
    "</think>\n\n<tool_call>\n"
    '{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['
)

_IMAGE_CONFIG: dict = {}

def prepare(processor, row: dict) -> dict[str, torch.Tensor]:
    """Run image and prompt preprocessing once for reusable output prefixes."""
    if not _IMAGE_CONFIG:
        image_processor = processor.image_processor
        _IMAGE_CONFIG.update(
            patch_size=image_processor.patch_size,
            merge_size=image_processor.merge_size,
            min_pixels=image_processor.size.get("shortest_edge"),
            max_pixels=image_processor.size.get("longest_edge"),
        )
    with Image.open(ROOT / row["crop"]) as source:
        crop = source.convert("RGB")
        prompt_info = get_computer_use_system_prompt(
            crop, _IMAGE_CONFIG, display_size=DISPLAY_SIZE
        )
        system_text = "".join(
            part.get("text", "") for part in prompt_info["conversation"][0]["content"]
        )
        resized = crop.resize(prompt_info["im_size"])
        history = ""
        if row["history"]:
            history = "\nCompleted actions so far:\n" + "\n".join(
                f"{index + 1}. {action}" for index, action in enumerate(row["history"])
            )
        prompt = processor.apply_chat_template(
            [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["task"] + history},
                    ],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        data = processor(text=[prompt], images=[resized], return_tensors="pt")

    return dict(data)


def encode_prepared(
    processor,
    prepared: dict[str, torch.Tensor],
    prefix_text: str,
) -> dict[str, torch.Tensor]:
    suffix = torch.tensor(
        [processor.tokenizer.encode(prefix_text, add_special_tokens=False)],
        dtype=torch.long,
    )
    data = dict(prepared)
    data["input_ids"] = torch.cat([prepared["input_ids"], suffix], 1)
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    if "mm_token_type_ids" in data:
        data["mm_token_type_ids"] = torch.cat(
            [prepared["mm_token_type_ids"], torch.zeros_like(suffix)], 1
        )
    return {key: value.cuda() for key, value in data.items()}


def encode(processor, row: dict, prefix_text: str) -> dict[str, torch.Tensor]:
    return encode_prepared(processor, prepare(processor, row), prefix_text)


