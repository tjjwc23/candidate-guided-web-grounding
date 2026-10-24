"""Intervention invariants that matter for interpreting the cached ablations."""

import math

import pytest
import torch

from grounding.element_scoring import LinearScorer as LinearReranker
from grounding.element_features import first_window_features
from grounding.element_features import FEATURE_NAMES as FEATURE_NAMES
from grounding.element_features import features as features
from grounding.element_scoring import predict as predict


def multi_example():
    row = dict(
        index=0,
        action_key="t_a",
        task_id="t",
        page_size=[100, 200],
        candidate_bboxes=[[0, 0, 100, 200], [20, 20, 10, 10], [20, 120, 10, 10]],
        windows=[
            dict(top=0, centres={"0": [500, 500], "1": [250, 250]}),
            dict(top=100, centres={"2": [250, 250]}),
        ],
    )
    points = dict(
        action_key="t_a",
        windows=[
            dict(
                top=0,
                crop_size=[100, 100],
                coordinate=[250, 250],
                log_probability=-1.0,
                token_ids=[1, 2],
            ),
            dict(
                top=100,
                crop_size=[100, 100],
                coordinate=[250, 250],
                log_probability=-2.0,
                token_ids=[1, 2],
            ),
        ],
    )
    return row, points


def test_window_intervention_rebuilds_prior_and_support_not_just_candidate_filter():
    row, points = multi_example()
    multi = features(row, points, [0.0, 0.0, 0.0])
    one = first_window_features(
        dict(row, labels=[False, False, True]), points, [0.0, 0.0, 0.0]
    )
    assert multi["candidate_ids"] == [0, 1, 2]
    assert one["candidate_ids"] == [0, 1]
    assert one["base_scores"] == pytest.approx([-1 - math.log(2)] * 2)
    idx = FEATURE_NAMES.index("window_logprior")
    assert one["features"][1][idx] == 0
    assert multi["features"][1][idx] == pytest.approx(math.log(2 / 3))
    support = FEATURE_NAMES.index("hit_window_fraction")
    assert multi["features"][1][support] == 0.5
    assert one["features"][1][support] == 1.0
    assert len(row["windows"]) == len(points["windows"]) == 2


def test_first_window_miss_uses_original_c50_semantic_fallback():
    row, points = multi_example()
    points["windows"][0]["coordinate"] = None
    one = first_window_features(row, points, [0.0, 1.0, 3.0])
    assert one["candidate_ids"] == []
    assert predict(LinearReranker(), [one], "cpu")[0]["t_a"] == 2


def test_linear_zero_is_fixed_fusion_and_can_change_candidate_choice():
    row, points = multi_example()
    data = features(row, points, [0.0, 0.0, 0.0])
    model = LinearReranker()
    assert sum(p.numel() for p in model.parameters()) == 18
    _, records = predict(model, [data], "cpu")
    assert list(records[0]["candidate_log_scores"].values()) == pytest.approx(
        data["base_scores"]
    )
    with torch.no_grad():
        model.correction.weight[0, FEATURE_NAMES.index("proposal_rank")] = 1000.0
    assert predict(model, [data], "cpu")[0]["t_a"] == 2
