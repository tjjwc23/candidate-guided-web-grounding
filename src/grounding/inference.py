"""Unified element reranking from label-free, multi-window coordinate evidence."""

import argparse
import json
from pathlib import Path

import torch

import artifacts, manifest
from artifacts import write_json
from data.element_inputs import load_evidence
from evaluation.grounding_metrics import element_report
from execution import experiment_status
from grounding.element_scoring import load_model, predict


def evidence_paths(split, reference=None):
    reference = manifest.load() if reference is None else reference
    return tuple(
        manifest.frozen_input(name, split, reference)
        for name in ("grounding_points", "grounding_semantics")
    )


def evaluate(
    output, split, *, evidence=None, geometry=None, reference=None, head_path=None
):
    """Pool all hit elements, score once, and select the highest-scoring element."""
    reference = manifest.load() if reference is None else reference
    head_path = Path(head_path) if head_path else manifest.element_head()
    if geometry is None:
        rows_path = manifest.frozen_input("anchored_rows", split, reference)
        baseline_path = manifest.frozen_input("grounding_baseline", split, reference)
        page_path = manifest.frozen_input("grounding_page_index", split, reference)
    else:
        rows_path, baseline_path, page_path = (
            geometry / name
            for name in ("rows.json", "baseline.json", "page_index.jsonl")
        )
    source = artifacts.read_json(rows_path)
    pages = {
        r["action_key"]: r["page_size"]
        for r in map(json.loads, page_path.read_text().splitlines())
    }
    rows = [
        dict(
            {k: v for k, v in row.items() if k != "labels"},
            page_size=pages[row["action_key"]],
        )
        for row in source
    ]
    points, semantics = evidence or evidence_paths(split, reference)
    dataset = load_evidence(rows, points, semantics)
    predictions, records = predict(load_model(head_path), dataset, "cpu")
    write_json(output / "records.json", records)
    report = element_report(source, predictions, artifacts.read_json(baseline_path))
    summary = dict(
        status="complete",
        split=split,
        method="unified_element",
        result=report,
        element_head_sha256=artifacts.file_digest(head_path),
        element_model_sha256=artifacts.file_digest(head_path.parent / "model.json"),
        evidence_paths=[str(points), str(semantics)],
        coordinate_inference=False,
        semantic_inference=False,
        training=False,
    )
    write_json(output / "summary.json", summary)
    experiment_status(output, "complete", split=split, EA=report["task_macro_EA"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["all", "test_task", "test_website", "test_domain"],
        default="all",
    )
    parser.add_argument("--config", type=Path, help="Frozen evidence/adapter manifest")
    parser.add_argument(
        "--head",
        type=Path,
        help="18D exported head.pt with model.json; default from configs/element.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = manifest.load(args.config)
    splits = list(reference["splits"]) if args.split == "all" else [args.split]
    summaries = {
        split: evaluate(
            args.output / split, split, reference=reference, head_path=args.head
        )
        for split in splits
    }
    write_json(
        args.output / "summary.json",
        dict(status="complete", method="unified_element", splits=summaries),
    )
