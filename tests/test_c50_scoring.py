"""Pure-logic tests for the C50 element-level scorer (no GPU, no model)."""

import json
import math


import grounding.evidence as evidence
import evaluation.grounding_metrics as metrics
import manifest as manifest  # noqa: E402


def test_row_weights_are_log_candidate_shares_of_the_row_windows():
    row = {"windows": [{"centres": {"0": [1, 1], "1": [2, 2], "2": [3, 3]}},
                       {"centres": {"3": [4, 4]}}]}
    weights = evidence.row_weights(row)
    assert abs(weights[0] - math.log(0.75)) < 1e-9
    assert abs(weights[1] - math.log(0.25)) < 1e-9


def test_element_report_uses_the_official_full_denominator(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([
        {"sample_id": "a", "annotation_id": "t1"},
        {"sample_id": "b", "annotation_id": "t1"},
        {"sample_id": "c", "annotation_id": "t2"}]))
    rows = [{"action_key": "a", "labels": [1, 0]}, {"action_key": "b", "labels": [1, 0]},
            {"action_key": "c", "labels": [1, 0]}]
    # a correct (1 of t1's two steps), b wrong, c correct -> macro = (0.5 + 1.0) / 2 = 0.75
    report = metrics.element_report(rows, {"a": 0, "b": 1, "c": 0}, json.loads(predictions.read_text()))
    assert abs(report["task_macro_EA"] - 0.75) < 1e-9
    assert report["eligible_micro_EA"] == 2 / 3 and report["tasks"] == 2


def test_element_report_counts_unscored_steps_as_zero(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([{"sample_id": "a", "annotation_id": "t1"},
                                       {"sample_id": "z", "annotation_id": "t1"}]))
    report = metrics.element_report([{"action_key": "a", "labels": [1]}], {"a": 0}, json.loads(predictions.read_text()))
    assert abs(report["task_macro_EA"] - 0.5) < 1e-9      # the unprepared step stays in the denominator
    assert report["steps_scored"] == 1


def test_domain_report_uses_raw_full_denominator_without_frozen_predictions(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Fresh split evaluation must not read reference predictions")

    monkeypatch.setattr(manifest, "predictions", forbidden)
    baseline = [{"sample_id": "a", "annotation_id": "long"},
                {"sample_id": "missing", "annotation_id": "long"},
                {"sample_id": "b", "annotation_id": "short"}]
    rows = [{"action_key": "a", "labels": [False] * 20 + [True]},
            {"action_key": "b", "labels": [True]}]
    report = metrics.element_report(rows, {"a": 20, "b": 0}, baseline)
    assert report["denominator_steps"] == 3 and report["tasks"] == 2
    assert report["correct_steps"] == 2 and report["micro_EA"] == 2 / 3
    assert report["task_macro_EA"] == .75


def test_task_pairing_weights_tasks_equally_and_preserves_fixes_breaks():
    from evaluation.grounding_metrics import paired
    baseline=[dict(sample_id='a',annotation_id='long'),dict(sample_id='b',annotation_id='long'),
              dict(sample_id='c',annotation_id='short')]
    result=paired(baseline,{'a':True,'b':True},{'b':True,'c':True})
    assert result['fixes']==result['breaks']==1
    assert result['delta_pp']==25.
    assert result['task_bootstrap95_pp']==[-50.,100.]
