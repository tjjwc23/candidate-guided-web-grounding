"""evaluation / proposals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import datasets.features.features as hf_features

from artifacts import write_json
from data.mind2web import TEST_SPLITS, action_key, load_action_records
from retrieval.inputs import (
    build_dom_candidate_sample,
    default_proposal_path,
    load_proposal_cache,
    proposal_indices,
)

DATASET = "osunlp/Multimodal-Mind2Web"

hf_features._FEATURE_TYPES.setdefault("List", hf_features.Sequence)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_proposal_rows(rows, *, split, top_k, source_top_k):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)

    total = len(rows)
    eligible_rows = [row for row in rows if row["proposal_available"]]
    recalled = sum(row["gold_recalled"] for row in rows)
    top1_correct = sum(row["top1_correct"] for row in rows)
    return {
        "metric_protocol": "mind2web-official-task-macro-v3",
        "method": "public_mindact_deberta",
        "split": split,
        "candidate_cutoff": top_k,
        "source_cache_top_k": source_top_k,
        "total_actions": total,
        "proposal_actions": len(eligible_rows),
        "missing_proposal_actions": total - len(eligible_rows),
        "task_count": len(by_task),
        "gold_recalled_actions": recalled,
        "top1_correct_actions": top1_correct,
        "proposal_availability": len(eligible_rows) / total if total else 0.0,
        "recall_at_k_full": recalled / total if total else 0.0,
        "recall_at_k_eligible": (
            recalled / len(eligible_rows) if eligible_rows else 0.0
        ),
        "top1_micro_ea_full": top1_correct / total if total else 0.0,
        "top1_micro_ea_eligible": (
            top1_correct / len(eligible_rows) if eligible_rows else 0.0
        ),
        "c10_oracle_task_macro_ea": _mean(
            [
                _mean([float(row["gold_recalled"]) for row in task_rows])
                for task_rows in by_task.values()
            ]
        ),
        "top1_task_macro_ea": _mean(
            [
                _mean([float(row["top1_correct"]) for row in task_rows])
                for task_rows in by_task.values()
            ]
        ),
        "missing_proposals_count_as_errors": True,
    }


def evaluate_split(*, split, data_path, proposals_path, top_k):
    proposal_cache = load_proposal_cache(proposals_path)
    source_top_k = int(proposal_cache["protocol"]["top_k"])
    if top_k > source_top_k:
        raise ValueError(f"Requested C{top_k} from a C{source_top_k} proposal cache")

    proposals = proposal_cache["actions"]
    records = load_action_records(data_path, split)
    if "screenshot" in records.column_names:
        records = records.remove_columns("screenshot")

    rows = []
    matched = 0
    for index in range(len(records)):
        record = records[index]
        key = action_key(record)
        proposal_available = key in proposals
        gold_recalled = False
        top1_correct = False
        if proposal_available:
            matched += 1
            sample = build_dom_candidate_sample(record)
            if sample is None:
                raise ValueError(f"Could not rebuild proposal candidates at {key}")
            positions = proposal_indices(proposal_cache, key, sample)[:top_k]
            if len(positions) != top_k:
                raise ValueError(f"Fewer than C{top_k} proposals at {key}")
            labels = [bool(sample["labels"][position]) for position in positions]
            gold_recalled = any(labels)
            top1_correct = labels[0]
        rows.append(
            {
                "task_id": str(record["annotation_id"]),
                "proposal_available": proposal_available,
                "gold_recalled": gold_recalled,
                "top1_correct": top1_correct,
            }
        )
        if (index + 1) % 500 == 0 or index + 1 == len(records):
            print(f"PROPOSAL_METRICS {split} {index + 1}/{len(records)}", flush=True)

    if matched != len(proposals):
        raise ValueError(f"Matched {matched} of {len(proposals)} proposal actions")
    return summarize_proposal_rows(
        rows,
        split=split,
        top_k=top_k,
        source_top_k=source_top_k,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=TEST_SPLITS)
    parser.add_argument("--data-path", default=DATASET)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proposals_path = args.proposals or default_proposal_path(args.split)
    report = evaluate_split(
        split=args.split,
        data_path=args.data_path,
        proposals_path=proposals_path,
        top_k=args.top_k,
    )
    write_json(args.output, report)
    print(
        f"PROPOSAL_METRICS complete | {args.split} | "
        f"R@{args.top_k} {100 * report['recall_at_k_full']:.2f}% | "
        f"Top-1 task-macro EA {100 * report['top1_task_macro_ea']:.2f}%",
        flush=True,
    )
