"""data / grounding inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from data.dom import build_candidate_ids
from data.mind2web import HISTORY_STEPS, action_key
from data.screenshots import (
    decode_screenshot,
    prepare_candidate_windows,
)
from retrieval.inputs import (
    build_dom_candidate_sample,
    default_proposal_path,
    load_proposal_cache,
    proposal_indices,
)

DATASET = "osunlp/Multimodal-Mind2Web"
DATASET_REVISION = "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889"


def raw_records(split, columns=None):
    """At most two HTML/screenshot rows live in each Arrow batch."""
    snapshot = Path(snapshot_download(repo_id=DATASET, revision=DATASET_REVISION, repo_type="dataset", local_files_only=True))
    for path in sorted((snapshot / "data").glob(f"{split}-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2, columns=columns):
            yield from batch.to_pylist()


def operation(record):
    value = record.get("operation") or {}
    return json.loads(value) if isinstance(value, str) else value


def query_history(record):
    previous = record.get("previous_actions")
    if previous is None:
        previous = (record.get("action_reprs") or [])[:int(record["target_action_index"])]
    return list(previous)


def select_tasks(identities, limit, seed=42):
    """Keep complete tasks; a sampling limit is a minimum number of actions."""
    if not limit:
        return {str(r["annotation_id"]) for r in identities}
    counts = Counter(str(r["annotation_id"]) for r in identities)
    ordered = sorted(counts, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    selected, count = set(), 0
    for task in ordered:
        selected.add(task)
        count += counts[task]
        if count >= limit:
            break
    return selected


def training_rows(rows):
    """Supervision selects a positive view after label-neutral window construction."""
    result = []
    for row in rows:
        selected = next((w for w in row["windows"] if any(
            row["labels"][int(i)] for i in w["centres"])), None)
        reason = "no_gold_official_c50" if not any(row["labels"]) else (
            "no_pos_window" if selected is None else None)
        window = None
        if selected is not None:
            centres = [selected["centres"].get(str(i)) for i in range(len(row["node_ids"]))]
            window = {**selected, "centres": centres,
                      "covered": [point is not None for point in centres]}
        result.append({**row, "annotation_id": row["task_id"], "window": window, "skip_reason": reason})
    return result


def prepare(split, output, limit=0, seed=42, window_candidates=20):
    output = Path(output).resolve()
    cache_path = default_proposal_path(split)
    proposals = load_proposal_cache(cache_path)
    identities = list(raw_records(split, ["annotation_id", "action_uid"]))
    tasks = select_tasks(identities, limit, seed)
    output.mkdir(parents=True, exist_ok=True)
    pages = output / "pages"
    pages.mkdir(exist_ok=True)
    rows, baseline, index_rows, skipped = [], [], [], Counter()
    for index, raw in enumerate(raw_records(split)):
        if str(raw["annotation_id"]) not in tasks:
            continue
        key, op = action_key(raw), operation(raw)
        raw["previous_actions"] = query_history(raw)
        entry = dict(sample_id=key, annotation_id=str(raw["annotation_id"]),
                     gold_action=op["op"], gold_value=str(op.get("value", "") or "").strip(),
                     eval_eligible=False, topk_hit=False, element_correct=False,
                     operation_f1=0., step_success=False)
        baseline.append(entry)
        if raw.get("screenshot") is None:
            skipped["missing_screenshot"] += 1
            continue
        with decode_screenshot(raw["screenshot"]) as page:
            page_path = pages / f"{key}.png"
            page.save(page_path)
            index_rows.append(dict(action_key=key, annotation_id=str(raw["annotation_id"]),
                                   action_uid=str(raw["action_uid"]), orig_size=list(page.size),
                                   page_size=list(page.size)))
            if key not in proposals["actions"]:
                skipped["no_public_proposal"] += 1
                continue
            sample = build_dom_candidate_sample(raw)
            if sample is None:
                skipped["invalid_candidate_sample"] += 1
                continue
            positions = proposal_indices(proposals, key, sample)[:50]
            boxes = [sample["candidate_bboxes"][i] for i in positions]
            windows = prepare_candidate_windows(
                page, dict(enumerate(boxes)), output, index,
                layout_boxes=dict(enumerate(boxes[:window_candidates])))
            if not windows:
                skipped["no_valid_geometry"] += 1
                continue
        ids = build_candidate_ids(sample["backend_node_ids"])
        labels = [bool(sample["labels"][i]) for i in positions]
        rows.append(dict(index=index, action_key=key, task_id=str(raw["annotation_id"]),
                         website=str(raw["website"]), task=str(raw["confirmed_task"]),
                         history=raw["previous_actions"][-HISTORY_STEPS:], node_ids=[ids[i] for i in positions],
                         candidate_bboxes=boxes, labels=labels, n_gold=sum(labels), windows=windows))
        entry.update(topk_hit=any(labels), eval_eligible=any(labels))
        if len(rows) % 100 == 0:
            print(json.dumps(dict(stage="prepare_c50", split=split, prepared=len(rows),
                                  denominator=len(baseline))), flush=True)
    for name, value in [("rows.json", rows), ("baseline.json", baseline)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False))
    (output / "page_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index_rows))
    summary = dict(split=split, denominator=len(baseline), tasks=len(tasks), prepared_rows=len(rows),
                   dataset_revision=DATASET_REVISION,
                   candidate_budget=50, candidate_count_policy="up_to_budget",
                   window_source=f"current_top{window_candidates}_candidates",
                   window_candidate_budget=window_candidates, labels_used_for_geometry=False,
                   proposal_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(), skipped=dict(skipped))
    (output / "preparation.json").write_text(json.dumps(summary, indent=2))
    if split == "train":
        train = training_rows(rows)
        (output / "train_rows.json").write_text(json.dumps(train, ensure_ascii=False))
        summary.update(partition=f"public DeBERTa C50 train; Top{window_candidates} layout; no validation split",
                       selected_actions=len(train), training_actions=sum(not r["skip_reason"] for r in train),
                       unusable_actions=sum(bool(r["skip_reason"]) for r in train), validation_actions=0,
                       candidate_inventory="official_deberta_c50", top_k=50)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "test_task", "test_website", "test_domain"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--window-candidates", type=int, choices=(20, 50), default=20)
    args = parser.parse_args()
    prepare(args.split, args.output, args.limit, window_candidates=args.window_candidates)
