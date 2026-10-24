"""Selection invariants shared by native and historical cached-head recipes."""

import torch

from evaluation.grounding_metrics import evaluate_validation
from grounding.element_training import fit_ranking_head


def test_training_filters_unsupported_rows_and_selects_earliest_full_denominator_tie(
    tmp_path,
):
    from grounding import element_scoring as scoring

    width = 18

    def row(index, partition="fit"):
        return dict(
            index=index,
            action_key=str(index),
            task_id=partition,
            partition=partition,
            candidate_ids=[0, 1],
            features=[[-1.0] * width, [1.0] * width],
            base_scores=[0.0, 0.0],
            fallback=1,
        )

    # Cross the batch boundary, and include rows with no supported training target.
    dataset = [row(i) for i in range(129)]
    dataset += [
        dict(row(129), candidate_ids=[1], features=[[1.0] * width], base_scores=[0.0]),
        dict(row(130), candidate_ids=[], features=[], base_scores=[]),
        row(131),
    ]
    dataset += [
        row(132, "valid"),
        dict(row(133, "valid"), candidate_ids=[], features=[], base_scores=[]),
    ]
    labels = {r["action_key"]: [False, True, False] for r in dataset}
    labels["131"] = [False, False, True]
    baseline = [
        dict(sample_id=k, annotation_id="valid") for k in ("132", "133", "missing")
    ]
    metric = lambda predictions, rows: evaluate_validation(
        predictions, rows, labels, baseline
    )
    model, result = fit_ranking_head(
        tmp_path,
        dataset,
        labels,
        scoring.LinearScorer,
        42,
        metric,
        learning_rates=(0.01, 0.03),
        epochs=2,
    )
    assert result["informative_fit_actions"] == 129
    assert result["fit_actions"] == 132
    assert result["selected"] == dict(lr=0.01, epoch=1, valid_task_macro_EA=2 / 3)
    assert result["runs"][0]["history"][0]["valid_task_macro_EA"] == 1 / 3
    assert result["validation_records"][1]["semantic_recovery"]
    assert scoring.predict(model, dataset[-2:], "cpu")[0] == {"132": 1, "133": 1}


def test_epoch_zero_remains_eligible_when_training_cannot_improve(tmp_path):
    from grounding.element_scoring import LinearScorer as LinearReranker

    row = dict(
        index=0,
        action_key="v",
        task_id="valid",
        partition="valid",
        candidate_ids=[],
        features=[],
        base_scores=[],
        fallback=0,
    )
    model, result = fit_ranking_head(
        tmp_path,
        [row],
        {"v": [True]},
        LinearReranker,
        42,
        lambda predictions, rows: float(predictions["v"] == 0),
        learning_rates=(0.01, 0.03),
        epochs=2,
    )
    assert result["selected"] == dict(lr=0.01, epoch=0, valid_task_macro_EA=1.0)
    assert result["informative_fit_actions"] == 0
    assert all(torch.count_nonzero(p) == 0 for p in model.parameters())
