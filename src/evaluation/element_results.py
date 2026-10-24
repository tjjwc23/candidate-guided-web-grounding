from __future__ import annotations

import manifest as config
from artifacts import write_json
from data.element_inputs import prepare_test_features
from evaluation.grounding_metrics import element_report
from execution import experiment_status as status
from grounding.element_features import FEATURE_NAMES as FEATURE_NAMES
from grounding.element_features import features as features
from grounding.element_scoring import load_model
from grounding.element_scoring import predict as predict


def evaluate_tests(output, training):
    reference = config.load()
    model = load_model(output / "head.pt")
    results = {}
    for split in reference["splits"]:
        folder = output / "unified_element" / split
        source, baseline, dataset = prepare_test_features(folder, split, reference)
        predictions, records = predict(model, dataset, "cpu")
        write_json(folder / "records.json", records)
        report = element_report(source, predictions, baseline)
        gold = {r["action_key"]: r["labels"] for r in source}
        oracle = {
            r["action_key"]: next(
                (i for i in r["candidate_ids"] if gold[r["action_key"]][i]), None
            )
            for r in dataset
        }
        results[split] = dict(
            unified_element=report,
            all_hit_oracle=element_report(source, oracle, baseline),
            no_hit_actions=sum(not r["candidate_ids"] for r in dataset),
        )
        status(output, "evaluated", split=split, element_EA=report["task_macro_EA"])
    write_json(
        output / "summary.json",
        dict(status="complete", training=training, splits=results),
    )
    status(output, "complete", summary=str(output / "summary.json"))




