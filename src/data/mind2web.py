"""data / mind2web."""

from __future__ import annotations

import glob
import logging
import os

HISTORY_STEPS = 4
TEST_SPLITS = ("test_task", "test_website", "test_domain")
ALL_SPLITS = ("train", *TEST_SPLITS)


def format_query(task, history, *, history_steps=HISTORY_STEPS):
    return f"task is: {task}\nPrevious actions: {'; '.join(history[-history_steps:])}"


logger = logging.getLogger(__name__)


def action_key(record: dict) -> str:
    """Return the stable Mind2Web action identity used across pipeline stages."""
    return f"{record.get('annotation_id', '')}_{record.get('action_uid', '')}"




def _hf_local_files_only() -> bool:
    return any(
        os.environ.get(name, "").upper() in {"1", "ON", "TRUE", "YES"}
        for name in ("HF_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    )


def set_previous_actions(sample: dict) -> dict:
    """Attach previous actions for already-flattened records."""
    if "previous_actions" in sample:
        return {"previous_actions": sample.get("previous_actions", []) or []}

    action_reprs = sample.get("action_reprs", []) or []
    target_idx = sample.get("target_action_index")
    if target_idx is not None:
        try:
            idx = int(target_idx)
        except (TypeError, ValueError):
            idx = len(action_reprs) - 1
    else:
        idx = len(action_reprs) - 1
    previous = action_reprs[:idx] if idx > 0 else []
    return {"previous_actions": previous}


def flatten_actions(samples):
    """Flatten Mind2Web JSON structure: Task -> Actions -> Pairs."""
    outputs = {
        "website": [],
        "confirmed_task": [],
        "annotation_id": [],
        "previous_actions": [],
        "action_uid": [],
        "operation": [],
        "pos_candidates": [],
        "neg_candidates": [],
        "cleaned_html": [],
    }

    num_actions = [len(actions) for actions in samples["actions"]]
    for key in ["website", "confirmed_task", "annotation_id"]:
        for idx, value in enumerate(samples[key]):
            outputs[key] += [value] * num_actions[idx]

    for actions, action_reprs in zip(samples["actions"], samples["action_reprs"]):
        for a_idx, action in enumerate(actions):
            outputs["previous_actions"].append(action_reprs[:a_idx])
            for key in [
                "action_uid",
                "operation",
                "pos_candidates",
                "neg_candidates",
                "cleaned_html",
            ]:
                outputs[key].append(action[key])

    return outputs


def _split_base_and_slice(dataset_split: str) -> tuple[str, str]:
    if "[" not in dataset_split:
        return dataset_split, ""
    base, suffix = dataset_split.split("[", 1)
    return base, "[" + suffix


def _load_parquet_fallback(data_path: str, dataset_split: str):
    from huggingface_hub import snapshot_download

    from datasets import load_dataset

    split_base, split_slice = _split_base_and_slice(dataset_split)
    local_dir = snapshot_download(
        repo_id=data_path,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_files_only=_hf_local_files_only(),
    )
    pattern = os.path.join(local_dir, "data", f"{split_base}-*.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet shards matched split={split_base!r} under {local_dir}"
        )
    logger.warning(
        "Falling back to parquet loading for %s split %s (%d shards)",
        data_path,
        dataset_split,
        len(parquet_files),
    )
    return load_dataset(
        "parquet",
        data_files={"data": parquet_files},
        split="data" + split_slice,
    )


def load_action_records(data_path: str, dataset_split: str):
    from datasets import load_dataset
    """Load dataset records with a stable flattened action schema."""
    if _hf_local_files_only():
        # Offline runs must not attempt the hub loader: it raises a hub-specific error that is not
        # the builtin ConnectionError, so the cached parquet shards are resolved directly instead.
        dataset = _load_parquet_fallback(data_path, dataset_split)
        return _flatten_records(dataset)
    try:
        dataset = load_dataset(
            data_path,
            split=dataset_split,
        )
    except ConnectionError:
        # In offline mode the Hub loader fails before it can discover cached
        # parquet shards. Resolve the cached snapshot explicitly instead.
        dataset = _load_parquet_fallback(data_path, dataset_split)
    except ValueError as exc:
        if "Feature type 'List' not found" not in str(exc):
            raise
        dataset = _load_parquet_fallback(data_path, dataset_split)
    return _flatten_records(dataset)


def _flatten_records(dataset):
    """Apply the same schema flattening to a hub-loaded or parquet-loaded dataset."""
    if "actions" in dataset.column_names:
        required_columns = [
            column
            for column in (
                "website",
                "confirmed_task",
                "annotation_id",
                "actions",
                "action_reprs",
            )
            if column in dataset.column_names
        ]
        if set(required_columns) != set(dataset.column_names):
            dataset = dataset.select_columns(required_columns)
        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names,
            "batch_size": 1,
        }
        map_kwargs["num_proc"] = 1
        return dataset.map(flatten_actions, **map_kwargs)

    return dataset.map(set_previous_actions)
