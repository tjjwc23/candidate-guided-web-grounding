"""fara."""

from __future__ import annotations

import os

import torch



def resolve_local_model(name: str | None, *, revision: str | None = None) -> str | None:
    if not name or os.path.isdir(name):
        return name
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=name,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        # Not cached locally (or online mode): keep the original identifier so
        # the caller's from_pretrained produces its normal error/download path.
        return name


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL = "microsoft/Fara1.5-4B"
BASE_REVISION = "776a33ae5b2ad503796a97ae20fdc66f61d2feea"


def load_model():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=True
    )
    kwargs = dict(
        revision=BASE_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_MODEL, **kwargs)
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = [
        n
        for n, _ in model.named_modules()
        if "language_model.layers." in n
        and n.split(".")[-1]
        in ["q_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"]
    ]
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.gradient_checkpointing_disable()
    model.eval()
    return processor, model








