"""Coordinate inference exports, independent of training checkpoint layouts."""
from pathlib import Path
from safetensors.torch import load_file
from model_artifacts import seal_export, read_export
from fara import BASE_MODEL, BASE_REVISION

SCORING_CONTRACT = "fara-coordinate-bf16-r16-history4-v1"
def export_model(directory, *, epoch):
    files = [str(p.relative_to(directory)) for p in (directory / "adapter").rglob("*") if p.is_file()]
    required = {"adapter/adapter_model.safetensors", "adapter/adapter_config.json"}
    if not required <= set(files):
        raise ValueError("Coordinate export requires adapter weights and config")
    return seal_export(directory, kind="coordinate", contract=SCORING_CONTRACT,
                       files=files, metadata=dict(epoch=epoch, base_model=BASE_MODEL,
                       base_revision=BASE_REVISION, history_steps=4))


def read_coordinate_model(directory):
    directory = Path(directory)
    value = read_export(directory, kind="coordinate", contract=SCORING_CONTRACT)
    if not {"adapter/adapter_model.safetensors", "adapter/adapter_config.json"} <= value["files"].keys():
        raise ValueError("Incomplete coordinate export")
    metadata = value["metadata"]
    if (metadata["base_model"], metadata["base_revision"], metadata["history_steps"]) != (BASE_MODEL, BASE_REVISION, 4):
        raise ValueError("Coordinate model contract changed")
    return value


def load_scoring_state(directory):
    directory = Path(directory)
    value = read_coordinate_model(directory)
    return dict(adapter=load_file(directory / "adapter/adapter_model.safetensors"), epoch=value["metadata"]["epoch"])
