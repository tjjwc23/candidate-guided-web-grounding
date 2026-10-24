import math

from evaluation.proposals import summarize_proposal_rows


def test_proposal_metrics_use_full_denominator_and_task_macro():
    rows = [
        {
            "task_id": "a",
            "proposal_available": True,
            "gold_recalled": True,
            "top1_correct": True,
        },
        {
            "task_id": "a",
            "proposal_available": True,
            "gold_recalled": True,
            "top1_correct": False,
        },
        {
            "task_id": "b",
            "proposal_available": False,
            "gold_recalled": False,
            "top1_correct": False,
        },
    ]
    report = summarize_proposal_rows(
        rows,
        split="test_task",
        top_k=10,
        source_top_k=50,
    )
    assert report["total_actions"] == 3
    assert report["proposal_actions"] == 2
    assert report["gold_recalled_actions"] == 2
    assert report["top1_correct_actions"] == 1
    assert math.isclose(report["recall_at_k_full"], 2 / 3)
    assert report["recall_at_k_eligible"] == 1.0
    assert math.isclose(report["top1_micro_ea_full"], 1 / 3)
    assert report["top1_micro_ea_eligible"] == 0.5
    assert report["c10_oracle_task_macro_ea"] == 0.5
    assert report["top1_task_macro_ea"] == 0.25
