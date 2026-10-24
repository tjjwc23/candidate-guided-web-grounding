"""grounding / coordinates."""

from __future__ import annotations

import re

import torch

import grounding.encoding as encoding
from grounding.encoding import A_PREFIX

COORDINATE_SUFFIX_TEXT = 'left_click", "coordinate": ['
assert A_PREFIX.endswith(COORDINATE_SUFFIX_TEXT)
COORDINATE_CACHE_PREFIX = A_PREFIX[:-len(COORDINATE_SUFFIX_TEXT)]

def coordinate_suffix_tokens(tokenizer):
    """Split the coordinate prefix while preserving the validated BF16 path."""
    cache_prefix = tokenizer.encode(COORDINATE_CACHE_PREFIX, add_special_tokens=False)
    coordinate_prefix = tokenizer.encode(A_PREFIX, add_special_tokens=False)
    if coordinate_prefix[: len(cache_prefix)] != cache_prefix:
        raise ValueError("Cache prefix is not a token prefix of coordinate inference")
    return coordinate_prefix[len(cache_prefix) :]


def load_state(*, checkpoint=None):
    """Return the processor, model and the tool-call suffix tokens."""
    from peft import set_peft_model_state_dict

    from fara import load_model
    from grounding.checkpoints import load_scoring_state
    from manifest import active_model

    assert torch.cuda.is_available(), "CUDA unavailable"
    state = load_scoring_state(checkpoint or active_model("coordinate"))
    processor, model = load_model()
    set_peft_model_state_dict(model, state["adapter"])
    del state
    model.eval()
    return {"processor": processor, "model": model,
            "suffix": coordinate_suffix_tokens(processor.tokenizer)}


def parse_coordinate_tail(text):
    # A single tokenizer token can contain the closing bracket plus JSON
    # closers. Stop at the bracket; do not reject an otherwise valid point.
    match = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not match:
        return None
    point = [int(value) for value in match.groups()]
    return point if all(0 <= value < encoding.DISPLAY_SIZE for value in point) else None




@torch.inference_mode()
def greedy_coordinate(processor, model, model_row, suffix, max_tokens=24):
    """Decode a coordinate with the training scorer’s split-prefix BF16 prefill."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        data = encoding.encode(processor, model_row, COORDINATE_CACHE_PREFIX)
        result = model(**data, use_cache=True, logits_to_keep=1)
        prefix_tokens = int(data["input_ids"].shape[1]) + len(suffix)
        result = model(
            input_ids=torch.tensor([suffix], device="cuda"),
            past_key_values=result.past_key_values, use_cache=True, logits_to_keep=1,
        )
        del data
        ids, log_probability, stopped = [], 0.0, False
        eos = processor.tokenizer.eos_token_id
        for _ in range(max_tokens):
            logits = result.logits[0, -1].float()
            token = int(logits.argmax())
            log_probability += float(logits[token] - torch.logsumexp(logits, -1))
            ids.append(token)
            text = processor.tokenizer.decode(ids, skip_special_tokens=False)
            if "]" in text or token == eos:
                stopped = True
                break
            result = model(
                input_ids=torch.tensor([[token]], device="cuda"),
                past_key_values=result.past_key_values, use_cache=True, logits_to_keep=1,
            )
        del result
    return {
        "tail": text, "token_ids": ids, "coordinate": parse_coordinate_tail(text),
        "log_probability": log_probability, "prefix_tokens": prefix_tokens,
        "stopped": stopped,
    }





