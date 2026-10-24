"""retrieval / inputs."""

from __future__ import annotations

import json
from pathlib import Path

from lxml import etree

import data.geometry as frame_geometry
import manifest as mainline
from data.dom import (
    build_dom_labeled_candidates,
    build_node_cache,
    candidate_order_identity,
    canonical_candidates,
)
from data.mind2web import HISTORY_STEPS, format_query, set_previous_actions

DEFAULT_PROPOSAL_DEBERTA_MODEL = "osunlp/MindAct_CandidateGeneration_deberta-v3-base"
PROPOSAL_CACHE_KIND = "reranker.deberta_proposals.v4"


def default_proposal_path(split, config=None):
    """Resolve the four-action proposal cache."""
    config = mainline.load() if config is None else config
    path = config.get('frozen_inputs', {}).get('public_proposals', {}).get(split)
    return mainline.resolve(path) if path else mainline.ROOT / f'outputs/mainline/inputs/{split}/proposals.json'




def load_proposal_cache(path: str | Path) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != PROPOSAL_CACHE_KIND:
        raise ValueError(f"Unsupported proposal cache: {payload.get('kind')!r}")
    return payload


def proposal_indices(cache: dict, action_key: str, sample=None) -> list[int]:
    row = cache["actions"].get(action_key)
    if row is None:
        raise KeyError(f"Proposal cache has no action {action_key!r}")
    if sample is not None and row.get(
        "candidate_order_identity"
    ) != candidate_order_identity(sample):
        raise ValueError(
            f"Proposal candidate identity/order changed for {action_key}; "
            "rebuild proposals"
        )
    return [int(index) for index in row["indices"]]


def _prepare_dom_candidates(record: dict):
    try:
        dom_tree = etree.fromstring(record.get("cleaned_html", ""))
    except Exception:
        return None, None
    node_cache = build_node_cache(dom_tree)
    candidates = build_dom_labeled_candidates(
        dom_tree,
        record.get("pos_candidates", []),
        record.get("neg_candidates", []),
        node_cache=node_cache,
    )
    if not candidates:
        return None, None

    candidates = canonical_candidates(candidates)

    # Mind2Web absolute boxes are CSS pixels local to their containing frame.
    # Project before downstream candidate geometry and image crops. An HTML root
    # rectangle may describe only the viewport, not the full screenshot size.
    candidates = frame_geometry.project_frame_candidates(
        candidates,
        node_cache,
        source_space=record.get("bbox_coordinate_space", "frame_local"),
    )
    return candidates, node_cache


def _candidate_sample(record: dict, candidates: list[dict], history_steps=HISTORY_STEPS) -> dict:
    dom_texts = [
        str(candidate.get("text", "") or "").strip() for candidate in candidates
    ]
    previous_actions = record.get("previous_actions")
    if previous_actions is None:
        previous_actions = set_previous_actions(record)["previous_actions"]
    return {
        "query": format_query(record.get('confirmed_task', ''), previous_actions or [], history_steps=history_steps),
        "dom_texts": dom_texts,
        "labels": [int(candidate["label"]) for candidate in candidates],
        "backend_node_ids": [
            candidate.get("backend_node_id", "") for candidate in candidates
        ],
        "candidate_bboxes": [candidate.get("bbox") for candidate in candidates],
        "candidate_bbox_normalized": [
            bool(candidate.get("bbox_normalized", False)) for candidate in candidates
        ],
        "candidate_geometry": [
            {
                key: candidate[key]
                for key in (
                    "bbox_source",
                    "bbox_space",
                    "bbox_frame_chain",
                    "bbox_projection_status",
                )
            }
            for candidate in candidates
        ],
        "document_size": None,
    }


def build_dom_candidate_sample(record: dict, *, history_steps=HISTORY_STEPS) -> dict | None:
    """Build the label-aligned DOM geometry used by public proposals."""
    candidates, _ = _prepare_dom_candidates(record)
    if candidates is None:
        return None
    sample = _candidate_sample(record, candidates, history_steps)
    sample["cand_texts"] = list(sample["dom_texts"])
    return sample
