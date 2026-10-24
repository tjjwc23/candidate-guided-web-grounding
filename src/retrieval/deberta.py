"""retrieval / deberta."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from lxml import etree
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from artifacts import write_json
from data.dom import candidate_order_identity
from data.mind2web import HISTORY_STEPS, load_action_records
from data.mind2web import action_key as make_action_key
from execution import set_seed
from fara import resolve_local_model
from retrieval.dom_pruning import (
    format_candidate as format_mindact_candidate,
)
from retrieval.inputs import (
    DEFAULT_PROPOSAL_DEBERTA_MODEL,
    PROPOSAL_CACHE_KIND,
    build_dom_candidate_sample,
)


def tokenize_candidates(tokenizer, batch: dict, max_length: int):
    """Use the public MindAct candidate/query pair order."""
    return tokenizer(
        batch["candidate_texts"],
        [batch["query"]] * len(batch["candidate_texts"]),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="osunlp/Multimodal-Mind2Web")
    parser.add_argument("--dataset_split", required=True)
    parser.add_argument("--deberta_model", default=DEFAULT_PROPOSAL_DEBERTA_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--encoder_batch_size", type=int, default=32)
    parser.add_argument("--preprocess_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-steps", type=int, choices=(4,), default=HISTORY_STEPS
    )
    return parser.parse_args()


def deduplicate_candidate_texts(candidate_texts):
    """Deduplicate documents while retaining the original candidate mapping."""
    unique_texts = []
    text_to_index = {}
    mapping = []
    for text in candidate_texts:
        if text not in text_to_index:
            text_to_index[text] = len(unique_texts)
            unique_texts.append(text)
        mapping.append(text_to_index[text])
    return unique_texts, mapping


def proposal_candidate_texts(record, prepared):
    """Build candidate texts in the exact order used by the DOM sample."""
    _, action_key, sample = prepared
    backend_node_ids = sample["backend_node_ids"]
    if len(backend_node_ids) != len(sample["labels"]):
        raise ValueError(f"Candidate count mismatch for {action_key}")
    return format_proposal_candidate_texts(record["cleaned_html"], backend_node_ids)


def format_proposal_candidate_texts(cleaned_html, backend_node_ids):
    """Format one action in a worker while preserving official serialization."""
    dom_tree = etree.fromstring(cleaned_html)
    return [
        format_mindact_candidate(
            dom_tree,
            {"backend_node_id": str(backend_node_id)},
        )
        for backend_node_id in backend_node_ids
    ]


def _iter_proposal_candidate_texts(records, prepared_records, worker_count):
    """Keep a bounded queue and yield CPU DOM-formatting jobs as they finish."""
    if worker_count <= 1:
        for prepared in prepared_records:
            yield prepared, proposal_candidate_texts(records[prepared[0]], prepared)
        return

    action_iterator = iter(prepared_records)
    pending = {}
    max_pending = worker_count * 2
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        for _ in range(max_pending):
            try:
                prepared = next(action_iterator)
            except StopIteration:
                break
            future = executor.submit(
                format_proposal_candidate_texts,
                records[prepared[0]]["cleaned_html"],
                prepared[2]["backend_node_ids"],
            )
            pending[future] = prepared

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                prepared = pending.pop(future)
                yield prepared, future.result()
                try:
                    next_prepared = next(action_iterator)
                except StopIteration:
                    continue
                next_future = executor.submit(
                    format_proposal_candidate_texts,
                    records[next_prepared[0]]["cleaned_html"],
                    next_prepared[2]["backend_node_ids"],
                )
                pending[next_future] = next_prepared


def score_candidates(model, tokenized, device, encoder_batch_size):
    token_types = getattr(tokenized, "token_type_ids", None)
    lengths = tokenized.attention_mask.sum(dim=1)
    order = torch.argsort(lengths, stable=True)
    scores = torch.empty(tokenized.input_ids.size(0), dtype=torch.float32)
    for start in range(0, order.numel(), encoder_batch_size):
        indices = order[start : start + encoder_batch_size]
        mask = tokenized.attention_mask.index_select(0, indices)
        width = int(mask.sum(dim=1).max())
        kwargs = {
            "input_ids": tokenized.input_ids.index_select(0, indices)[:, :width].to(
                device
            ),
            "attention_mask": mask[:, :width].to(device),
        }
        if token_types is not None:
            kwargs["token_type_ids"] = token_types.index_select(0, indices)[
                :, :width
            ].to(device)
        chunk_scores = model(**kwargs).logits.squeeze(-1).float().cpu()
        scores.index_copy_(0, indices, chunk_scores)
    return scores


def build_streaming_cache(output, split):
    """Rebuild complete official proposals without loading screenshots or all DOMs."""
    import gc
    import manifest
    from artifacts import read_json, file_digest
    from data.dom import candidate_order_identity
    from data.grounding_inputs import raw_records, DATASET, DATASET_REVISION
    from execution import experiment_status

    model_path = resolve_local_model(DEFAULT_PROPOSAL_DEBERTA_MODEL)
    weight_hash = file_digest(Path(model_path) / "pytorch_model.bin")
    assert weight_hash == manifest.weight("deberta_proposals")["sha256"]
    protocol = dict(
        data_path=DATASET,
        dataset_split=split,
        dataset_revision=DATASET_REVISION,
        deberta_model=model_path,
        model_sha256=weight_hash,
        top_k=50,
        max_length=512,
        history_steps=4,
        scoring="public_initialization_once",
        candidate_text="dom_only",
        candidate_formatter="mindact_official",
        pair_order="candidate_query",
        precision="float32",
        deduplicate_candidate_texts=True,
        tie_break="stable_on_label_independent_candidate_order",
    )
    marker = output / "proposal_protocol.json"
    if marker.exists():
        assert read_json(marker) == protocol
    else:
        write_json(marker, protocol)
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=False)
    tokenizer.model_max_length = 512
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        .cuda()
        .eval()
    )
    columns = [
        "annotation_id",
        "action_uid",
        "confirmed_task",
        "action_reprs",
        "target_action_index",
        "cleaned_html",
        "pos_candidates",
        "neg_candidates",
    ]
    rows = {}
    total = sum(1 for _ in raw_records(split, ["annotation_id"]))
    with torch.inference_mode():
        for number, raw in enumerate(raw_records(split, columns), 1):
            key = make_action_key(raw)
            saved = output / "proposal_records" / f"{key}.json"
            if saved.exists():
                record = read_json(saved)
                assert record["action_key"] == key
            else:
                sample = build_dom_candidate_sample(raw, history_steps=4)
                record = dict(action_key=key, proposal=None)
                if sample is not None:
                    texts = format_proposal_candidate_texts(
                        raw["cleaned_html"], sample["backend_node_ids"]
                    )
                    unique, mapping = deduplicate_candidate_texts(texts)
                    tokens = tokenize_candidates(
                        tokenizer,
                        dict(query=sample["query"], candidate_texts=unique),
                        512,
                    )
                    unique_scores = score_candidates(
                        model, tokens, torch.device("cuda"), 16
                    )
                    scores = unique_scores[torch.tensor(mapping, dtype=torch.long)]
                    indices = np.argsort(-scores.numpy(), kind="stable")[:50].tolist()
                    record["proposal"] = dict(
                        candidate_order_identity=candidate_order_identity(sample),
                        candidate_count=len(sample["labels"]),
                        indices=indices,
                        logits=scores[indices].tolist(),
                    )
                write_json(saved, record)
            if record["proposal"] is not None:
                rows[key] = record["proposal"]
            if number == 1 or number % 50 == 0 or number == total:
                experiment_status(
                    output,
                    "building_full_proposals",
                    split=split,
                    completed=number,
                    total=total,
                )
    write_json(
        output / "proposals.json",
        dict(
            schema_version=4,
            kind=PROPOSAL_CACHE_KIND,
            protocol=protocol,
            actions=rows,
            metrics=dict(raw_actions=total, actions=len(rows)),
        ),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if (
        min(
            args.top_k,
            args.max_length,
            args.encoder_batch_size,
            args.preprocess_workers,
        )
        <= 0
    ):
        raise ValueError("Proposal sizes must be positive")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("DeBERTa proposal generation requires CUDA")
    set_seed(args.seed)

    records = load_action_records(args.data_path, args.dataset_split)
    prepared_records = []
    for record_index, record in enumerate(records):
        sample = build_dom_candidate_sample(record, history_steps=args.history_steps)
        if sample is None:
            continue
        prepared_records.append((record_index, make_action_key(record), sample))

    formatted_records = _iter_proposal_candidate_texts(
        records, prepared_records, args.preprocess_workers
    )
    first_formatted_record = next(formatted_records, None)

    resolved_model = resolve_local_model(args.deberta_model)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, fix_mistral_regex=False)
    tokenizer.model_max_length = args.max_length
    model = (
        AutoModelForSequenceClassification.from_pretrained(resolved_model, num_labels=1)
        .to(device)
        .eval()
    )

    rows = {}
    gold_recalled = 0
    with torch.inference_mode():
        formatted_iterator = formatted_records
        if first_formatted_record is not None:
            formatted_iterator = chain((first_formatted_record,), formatted_records)
        for prepared, candidate_texts in tqdm(
            formatted_iterator,
            total=len(prepared_records),
            desc=f"Build Top-{args.top_k} proposals",
        ):
            _, action_key, sample = prepared
            unique_texts, candidate_mapping = deduplicate_candidate_texts(
                candidate_texts
            )
            scoring_batch = {
                "query": sample["query"],
                "candidate_texts": unique_texts,
            }
            tokenized = tokenize_candidates(tokenizer, scoring_batch, args.max_length)
            unique_scores = score_candidates(
                model, tokenized, device, args.encoder_batch_size
            )
            scores = unique_scores[torch.tensor(candidate_mapping, dtype=torch.long)]
            count = min(args.top_k, scores.numel())
            indices = np.argsort(-scores.numpy(), kind="stable")[:count].tolist()
            gold_recalled += int(any(sample["labels"][index] for index in indices))
            rows[action_key] = {
                "candidate_order_identity": candidate_order_identity(sample),
                "indices": indices,
                "candidate_count": len(sample["labels"]),
            }

    output = Path(args.output).resolve()
    payload = {
        "schema_version": 4,
        "kind": PROPOSAL_CACHE_KIND,
        "protocol": {
            "data_path": args.data_path,
            "dataset_split": args.dataset_split,
            "deberta_model": str(resolved_model),
            "top_k": args.top_k,
            "max_length": args.max_length,
            "history_steps": args.history_steps,
            "scoring": "public_initialization_once",
            "candidate_text": "dom_only",
            "candidate_formatter": "mindact_official",
            "pair_order": "candidate_query",
            "deduplicate_candidate_texts": True,
            "preprocess_workers": args.preprocess_workers,
            "precision": "float32",
            "tie_break": "stable_on_label_independent_candidate_order",
            "input_eligibility": "valid_candidate_inventory_independent_of_labels",
        },
        "metrics": {
            "actions": len(prepared_records),
            "gold_recalled": gold_recalled,
            "gold_recall_at_k": gold_recalled / max(1, len(prepared_records)),
        },
        "actions": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(json.dumps({"output": str(output), **payload["metrics"]}, ensure_ascii=False))
