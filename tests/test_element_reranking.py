"""The direct comparator retains overlapping boxes, uses no targets, and shares evidence fairly."""


import copy
import math

import pytest
import torch

from grounding.element_features import FEATURE_NAMES as FEATURE_NAMES
from grounding.element_scoring import ElementScorer
from grounding.element_scoring import collate as collate
from grounding.element_features import features as features
from grounding.element_scoring import predict as predict
from grounding.element_scoring import ranking_loss


def example():
    row = dict(
        index=0,
        action_key="task_action",
        task_id="task",
        page_size=[100, 100],
        candidate_bboxes=[[0, 0, 100, 100], [20, 20, 10, 10], [70, 70, 10, 10]],
        windows=[
            dict(top=0, centres={"0": [500, 500], "1": [250, 250], "2": [750, 750]})
        ],
    )
    points = dict(
        action_key=row["action_key"],
        windows=[
            dict(
                top=0,
                crop_size=[100, 100],
                coordinate=[250, 250],
                log_probability=-1.0,
                token_ids=[1, 2],
            )
        ],
    )
    return row, points


def test_all_hit_parent_can_win_without_local_gate_or_label_input():
    row, points = example()
    a = features(dict(row, labels=[False, True, False]), points, [3.0, -3.0, 10.0])
    b = features(dict(row, labels=[True, False, False]), points, [3.0, -3.0, 10.0])
    assert a == b
    assert a["candidate_ids"] == [0, 1]
    model = ElementScorer().eval()
    predictions, records = predict(model, [a], "cpu")
    assert predictions["task_action"] == 0
    # The highest-semantic non-hit candidate is ineligible while hit candidates exist.
    assert "2" not in records[0]["candidate_log_scores"]


def test_no_hit_semantic_recovery_and_padding_do_not_change_other_actions():
    row, points = example()
    regular = features(row, points, [0.0, 1.0, -1.0])
    no_hit = copy.deepcopy(points)
    no_hit["windows"][0]["coordinate"] = None
    empty = features(
        dict(row, action_key="empty"), dict(no_hit, action_key="empty"), [0.0, 2.0, 2.0]
    )
    model = ElementScorer().eval()
    alone, _ = predict(model, [regular], "cpu")
    together, records = predict(model, [regular, empty], "cpu")
    assert alone["task_action"] == together["task_action"]
    assert together["empty"] == 1
    assert records[1]["semantic_recovery"]


def test_repeated_equal_windows_preserve_aggregate_under_normalized_prior():
    row, points = example()
    one = features(row, points, [0.0, 0.0, 0.0])
    twice_row, twice_points = copy.deepcopy(row), copy.deepcopy(points)
    twice_row["windows"] *= 2
    twice_points["windows"] *= 2
    two = features(twice_row, twice_points, [0.0, 0.0, 0.0])
    assert two["candidate_ids"] == one["candidate_ids"]
    assert two["base_scores"] == pytest.approx(one["base_scores"])
    assert one["base_scores"] == pytest.approx([-1.0 - math.log(2.0)] * 2)


def test_multiple_positive_objective_and_gradient_can_promote_larger_box():
    row, points = example()
    data = features(row, points, [-2.0, 2.0, 0.0])
    torch.manual_seed(20261012)
    model = ElementScorer()
    both = collate([data], "cpu", {"task_action": [True, True, False]}, {"task": 1})
    assert float(ranking_loss(model, both).detach()) == pytest.approx(0.0)
    batch = collate([data], "cpu", {"task_action": [True, False, False]}, {"task": 1})
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = float(ranking_loss(model, batch).detach())
    for _ in range(30):
        optimizer.zero_grad()
        loss = ranking_loss(model, batch)
        loss.backward()
        optimizer.step()
    assert float(ranking_loss(model, batch).detach()) < initial / 10
    assert predict(model.eval(), [data], "cpu")[0]["task_action"] == 0


def test_mismatched_window_cache_is_rejected():
    row, points = example()
    points["windows"][0]["top"] = 10
    with pytest.raises(AssertionError):
        features(row, points, [0.0, 1.0, 2.0])


def test_batch_boundary_ties_and_all_empty_fallback():
    model = ElementScorer().eval()
    rows = [
        dict(
            index=i,
            action_key=str(i),
            task_id="task",
            candidate_ids=[3, 1],
            features=[[0.0] * len(FEATURE_NAMES)] * 2,
            base_scores=[0.0, 0.0],
            fallback=7,
        )
        for i in range(129)
    ]
    predictions, records = predict(model, rows, "cpu")
    assert len(records) == 129 and set(predictions.values()) == {1}
    empty = [dict(row, candidate_ids=[], features=[], base_scores=[]) for row in rows]
    assert collate(empty, "cpu")["features"].shape == (129, 1, len(FEATURE_NAMES))
    predictions, records = predict(model, empty, "cpu")
    assert set(predictions.values()) == {7}
    assert all(
        r["semantic_recovery"] and r["candidate_log_scores"] == {} for r in records
    )
