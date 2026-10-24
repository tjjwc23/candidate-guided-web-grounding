import json

import pytest
from lxml import etree

from data.dom import build_node_cache
from data.geometry import project_frame_candidates
from retrieval.inputs import build_dom_candidate_sample
from retrieval.inputs import PROPOSAL_CACHE_KIND; from retrieval.inputs import load_proposal_cache
from retrieval.deberta import tokenize_candidates


def _record():
    return {
        "annotation_id": "annotation",
        "action_uid": "action",
        "confirmed_task": "Submit the form",
        "previous_actions": ["TYPE email"],
        "cleaned_html": (
            '<html backend_node_id="1" bounding_box_rect="0,0,1000,800">'
            '<body backend_node_id="2"><button backend_node_id="42" '
            'bounding_box_rect="100,80,100,80">Submit</button></body></html>'
        ),
        "pos_candidates": [
            {
                "backend_node_id": "42",
                "text": "Submit",
                "bbox": [100, 80, 100, 80],
                "bbox_normalized": False,
            }
        ],
        "neg_candidates": [],
    }


def test_direct_dom_sample_has_aligned_geometry_and_query():
    sample = build_dom_candidate_sample(_record())
    assert sample["labels"] == [1]
    assert sample["backend_node_ids"] == ["42"]
    assert sample["candidate_bboxes"] == [[100, 80, 100, 80]]
    assert sample["cand_texts"] == sample["dom_texts"]
    assert "Submit the form" in sample["query"]


def test_pair_order_is_candidate_then_query():
    calls = []

    class RecordingTokenizer:
        def __call__(self, first, second, **kwargs):
            calls.append((first, second, kwargs))
            return "tokenized"

    batch = {"candidate_texts": ["candidate-a", "candidate-b"], "query": "task"}
    assert tokenize_candidates(RecordingTokenizer(), batch, 512) == "tokenized"
    assert calls[0][0] == ["candidate-a", "candidate-b"]
    assert calls[0][1] == ["task", "task"]


def test_only_current_proposal_cache_contract_is_accepted(tmp_path):
    current = tmp_path / "current.json"
    current.write_text(json.dumps({"kind": PROPOSAL_CACHE_KIND, "actions": {}}))
    assert load_proposal_cache(current)["kind"] == PROPOSAL_CACHE_KIND

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"kind": "reranker.deberta_proposals.v1"}))
    with pytest.raises(ValueError, match="Unsupported proposal cache"):
        load_proposal_cache(legacy)


def _frame_fixture():
    tree = etree.fromstring(
        '<html><iframe backend_node_id="outer">'
        '<iframe backend_node_id="inner">'
        '<input backend_node_id="field" placeholder="Zip"/>'
        "</iframe></iframe></html>"
    )
    candidates = [
        {"backend_node_id": "field", "bbox": [5, 6, 30, 20]},
        {"backend_node_id": "outer", "bbox": [100, 200, 400, 300]},
        {"backend_node_id": "inner", "bbox": [10, 20, 200, 100]},
    ]
    return candidates, build_node_cache(tree)


def test_nested_frames_are_order_independent_and_idempotent():
    candidates, nodes = _frame_fixture()
    projected = project_frame_candidates(candidates, nodes, source_space="frame_local")
    assert projected[0]["bbox"] == [115, 226, 30, 20]
    assert projected[2]["bbox"] == [110, 220, 200, 100]
    assert candidates[0]["bbox"] == [5, 6, 30, 20]
    reverse = project_frame_candidates(
        candidates[::-1], nodes, source_space="frame_local"
    )
    assert {row["backend_node_id"]: row["bbox"] for row in projected} == {
        row["backend_node_id"]: row["bbox"] for row in reverse
    }
    assert project_frame_candidates(projected, nodes, source_space="frame_local") == projected


def test_explicit_global_boxes_are_not_shifted():
    candidates, nodes = _frame_fixture()
    assert project_frame_candidates(candidates, nodes, source_space="page")[0]["bbox"] == [
        5,
        6,
        30,
        20,
    ]


def test_unresolved_frame_is_not_treated_as_page_pixels():
    candidates, nodes = _frame_fixture()
    projected = project_frame_candidates(candidates[:1], nodes, source_space="frame_local")
    assert projected[0]["bbox"] is None
    assert projected[0]["bbox_projection_status"] == "unresolved_frame"


def test_projection_precedes_crop_and_ignores_root_viewport():
    record = {
        "annotation_id": "a",
        "action_uid": "b",
        "confirmed_task": "Fill Zip",
        "previous_actions": [],
        "cleaned_html": '<html backend_node_id="root" bounding_box_rect="0,0,1280,1080">'
        '<iframe backend_node_id="f"><input backend_node_id="z" placeholder="Zip"/>'
        "</iframe></html>",
        "pos_candidates": [
            {"backend_node_id": "z", "tag": "input", "bbox": [10, 20, 100, 40]}
        ],
        "neg_candidates": [
            {"backend_node_id": "f", "tag": "iframe", "bbox": [128, 208, 1024, 635]}
        ],
    }
    sample = build_dom_candidate_sample(record)
    index = sample["backend_node_ids"].index("z")
    assert sample["candidate_bboxes"][index] == [138, 228, 100, 40]
    global_sample = build_dom_candidate_sample(dict(record, bbox_coordinate_space="page"))
    assert global_sample["candidate_bboxes"][
        global_sample["backend_node_ids"].index("z")
    ] == [10, 20, 100, 40]
