from __future__ import annotations

import csv
import statistics
import time
from collections import Counter

import numpy as np

import manifest as config
from artifacts import write_json
from data.element_inputs import inputs, prepare_single_window
from evaluation.grounding_metrics import element_report, paired
from execution import experiment_status as status
from grounding.element_scoring import predict as predict


def evaluate_split(split, output, reference, models):
    rows, baseline, multi, single, original_records = prepare_single_window(
        split, output, reference
    )
    row_map = {r["action_key"]: r for r in rows}
    datasets = dict(multi_mlp=multi, single_mlp=single, multi_linear=multi)
    model_map = dict(
        multi_mlp=models["original"],
        single_mlp=models["original"],
        multi_linear=models["linear"],
    )
    reports = {}
    correct = {}
    predictions = {}
    for name, dataset in datasets.items():
        pred, records = predict(model_map[name], dataset, "cpu")
        write_json(output / split / f"{name}_records.json", records)
        reports[name] = element_report(rows, pred, baseline=baseline)
        correct[name] = {k: bool(row_map[k]["labels"][i]) for k, i in pred.items()}
        predictions[name] = pred
    assert predictions["multi_mlp"] == {
        r["action_key"]: r["winner"] for r in original_records
    }
    comparisons = {
        f"multi_mlp_minus_{name}": paired(baseline, correct[name], correct["multi_mlp"])
        for name in ("single_mlp", "multi_linear")
    }
    single_keys = {r["action_key"] for r in rows if len(r["windows"]) == 1}
    assert all(
        predictions["single_mlp"][k] == predictions["multi_mlp"][k] for k in single_keys
    )
    result = dict(
        methods=reports,
        comparisons=comparisons,
        window_counts=dict(sorted(Counter(len(r["windows"]) for r in rows).items())),
        no_hit_single=sum(not r["candidate_ids"] for r in single),
        no_hit_multi=sum(not r["candidate_ids"] for r in multi),
        changed_single_vs_multi=sum(
            predictions["single_mlp"][k] != predictions["multi_mlp"][k] for k in row_map
        ),
        unchanged_original_single_window_actions=len(single_keys),
    )
    write_json(output / split / "summary.json", result)
    status(
        output,
        "evaluated",
        split=split,
        EA={k: 100 * v["task_macro_EA"] for k, v in reports.items()},
        comparisons=comparisons,
    )
    return result


def evaluate_features(out, models, training, started):
    results = {}
    reference = config.load()
    for split in ("valid", *reference["splits"]):
        rows, baseline, data, original_records, _, _ = inputs(split, reference)
        labels = {r["action_key"]: r["labels"] for r in rows}
        tasks = sorted({b["annotation_id"] for b in baseline})
        ti = {t: i for i, t in enumerate(tasks)}
        counts = Counter(b["annotation_id"] for b in baseline)
        scores = {}
        reports = {}
        for name, by_seed in models.items():
            scores[name] = []
            reports[name] = {}
            for seed, model in by_seed.items():
                pred, records = predict(model.eval(), data, "cpu")
                write_json(out / name / str(seed) / split / "records.json", records)
                report = element_report(rows, pred, baseline=baseline)
                reports[name][str(seed)] = report
                task_scores = np.zeros(len(tasks))
                for b in baseline:
                    k = b["sample_id"]
                    t = b["annotation_id"]
                    if k in pred and labels[k][pred[k]]:
                        task_scores[ti[t]] += 1 / counts[t]
                assert abs(task_scores.mean() - report["task_macro_EA"]) < 1e-12
                scores[name].append(task_scores)
        rng = np.random.default_rng(26092026)
        indices = rng.integers(len(tasks), size=(10000, len(tasks)), dtype=np.int32)
        full = np.array(scores["full"])
        result = {}
        for name, per_seed in scores.items():
            a = np.array(per_seed)
            ea = a.mean(1) * 100
            diff = a - full
            average_diff = diff.mean(0)
            interval = np.quantile(
                average_diff[indices].mean(1) * 100, [0.025, 0.975]
            ).tolist()
            result[name] = dict(
                mean_EA_percent=float(ea.mean()),
                sample_sd_pp=float(ea.std(ddof=1)),
                delta_vs_full_pp=float(average_diff.mean() * 100),
                seed_deltas_pp=(diff.mean(1) * 100).tolist(),
                task_bootstrap95_mean_seed_delta_pp=interval,
                methods=reports[name],
            )
        results[split] = result
        write_json(out / split / "summary.json", result)
        status(out, "split_evaluated", split=split, seconds=time.monotonic() - started)
    summary = dict(
        status="complete",
        splits=results,
        training=training,
        seconds=time.monotonic() - started,
    )
    write_json(out / "summary.json", summary)
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "variant",
                "split",
                "mean_EA_percent",
                "sample_sd_pp",
                "delta_vs_full_pp",
                "ci_low",
                "ci_high",
            ]
        )
        for split, values in results.items():
            for name, r in values.items():
                writer.writerow(
                    [
                        name,
                        split,
                        r["mean_EA_percent"],
                        r["sample_sd_pp"],
                        r["delta_vs_full_pp"],
                        *r["task_bootstrap95_mean_seed_delta_pp"],
                    ]
                )
    write_json(
        out / "verification.json",
        dict(
            status="passed",
            native18_training=True,
            independently_aggregated_task_scores=True,
        ),
    )
    status(out, "complete", seconds=time.monotonic() - started)


def evaluate_seeds(out, models, training, started):
    results = {}
    reference = config.load()
    for split in ("valid", *reference["splits"]):
        rows, baseline, features, original, _, _ = inputs(split, reference)
        original_predictions = {r["action_key"]: r["winner"] for r in original}
        reports = {}
        changed = {}
        for seed, model in models.items():
            pred, records = predict(model, features, "cpu")
            reports[seed] = element_report(rows, pred, baseline=baseline)
            write_json(out / str(seed) / split / "records.json", records)
            changed[seed] = sum(pred[k] != v for k, v in original_predictions.items())
        values = [100 * r["task_macro_EA"] for r in reports.values()]
        results[split] = dict(
            methods=reports,
            mean_EA_percent=statistics.mean(values),
            sample_sd_pp=statistics.stdev(values),
            min_EA_percent=min(values),
            max_EA_percent=max(values),
            changed_vs_frozen=changed,
        )
        status(
            out,
            "evaluated",
            split=split,
            EA=values,
            mean=results[split]["mean_EA_percent"],
            sd=results[split]["sample_sd_pp"],
        )
    write_json(
        out / "summary.json",
        dict(
            status="complete",
            training=training,
            splits=results,
            seconds=time.monotonic() - started,
        ),
    )
    status(out, "complete", seconds=time.monotonic() - started)


def evaluate_controls(out, reference, models, training, started):
    splits = {
        s: evaluate_split(s, out, reference, models)
        for s in ("valid", *reference["splits"])
    }
    write_json(
        out / "summary.json",
        dict(
            status="complete",
            training=training,
            splits=splits,
            seconds=time.monotonic() - started,
        ),
    )
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "split",
                "method",
                "full_task_macro_EA_percent",
                "correct_actions",
                "denominator",
            ]
        )
        for split, result in splits.items():
            for name, r in result["methods"].items():
                writer.writerow(
                    [
                        split,
                        name,
                        100 * r["task_macro_EA"],
                        r["correct_steps"],
                        r["denominator_steps"],
                    ]
                )
    write_json(
        out / "verification.json",
        dict(
            status="passed",
            full_feature_replay=True,
            frozen_head_prediction_replay=True,
            original_single_window_invariance=True,
            actions_checked=sum(
                v["methods"]["multi_mlp"]["steps_scored"] for v in splits.values()
            ),
            note="Exact source-feature and original prediction replay plus singleton-window invariance; independent metric replay is separate",
        ),
    )
    status(out, "complete", seconds=time.monotonic() - started)
