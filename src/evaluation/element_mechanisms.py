from __future__ import annotations

import csv
import json
import math
import time

import numpy as np

import artifacts as artifacts
from artifacts import write_json
from data.element_inputs import (
    FEATURES,
    PREDICTIONS,
    SEEDS,
    evidence_rows,
    full_candidate_recall,
    inputs,
)
from evaluation.grounding_metrics import (
    Metrics,
    correctness,
    coverage_group,
    task_macro_weights,
    element_report,
    local_group,
    paired,
)
from execution import experiment_status as status
from grounding.element_features import restrict_candidates
from grounding.element_scoring import predict as predict
from grounding.evidence import choose_scores, fixed_scores, packed_evidence
from grounding.evidence import proposal_info


def classify(by_window):
    union = set().union(*by_window)
    if not union:
        return "no_hit"
    if len(union) == 1:
        return "single_candidate"
    if any(len(ids) > 1 for ids in by_window):
        return "overlap"
    return "cross_window_only"


def analyze(split, output, reference):
    rows, baseline, features, learned, point_dir, _ = inputs(split, reference)
    by_key = {r["action_key"]: r for r in rows}
    feature_map = {r["action_key"]: r for r in features}
    unified_element = {r["action_key"]: r["winner"] for r in learned}
    assert len(by_key) == len(rows) == len(features) == len(learned)
    assert set(by_key) == set(feature_map) == set(unified_element)
    predictions = dict(unified_element=unified_element, fixed_fusion={}, semantic_only={})
    strata = {r["sample_id"]: "unprepared" for r in baseline}
    diagnostics = {}
    for key, row in by_key.items():
        f = feature_map[key]
        ids = f["candidate_ids"]
        for name, values in (
            ("fixed_fusion", f["base_scores"]),
            ("semantic_only", [v[0] for v in f["features"]]),
        ):
            predictions[name][key] = (
                ids[max(range(len(ids)), key=lambda j: (values[j], -ids[j]))]
                if ids
                else f["fallback"]
            )
        points = artifacts.read_json(point_dir / f"{row['index']}.json")
        assert points["action_key"] == key
        hits, native, _ = proposal_info(row, points)
        assert sorted(hits) == ids
        by_window = [
            {
                i
                for i, support in hits.items()
                if any(h["window"] == wi for h in support)
            }
            for wi in range(len(points["windows"]))
        ]
        strata[key] = classify(by_window)
        diagnostics[key] = dict(
            hit_ids=ids,
            hit_ids_by_window=[sorted(s) for s in by_window],
            smallest_hit_ids=sorted(native),
        )
        if strata[key] in ("no_hit", "single_candidate"):
            assert len({p[key] for p in predictions.values()}) == 1
    correct = {
        name: {k: bool(by_key[k]["labels"][i]) for k, i in pred.items()}
        for name, pred in predictions.items()
    }
    reports = {
        name: element_report(rows, pred, baseline=baseline)
        for name, pred in predictions.items()
    }
    comparisons = {
        f"unified_element_minus_{name}": paired(baseline, values, correct["unified_element"])
        for name, values in correct.items()
        if name != "unified_element"
    }
    weights, _, _ = task_macro_weights(baseline=baseline)
    records = []
    for b in baseline:
        key = b["sample_id"]
        records.append(
            dict(
                action_key=key,
                task_id=b["annotation_id"],
                group=strata[key],
                winners={name: p.get(key) for name, p in predictions.items()},
                correct={name: p.get(key, False) for name, p in correct.items()},
                task_macro_weight=weights[key],
                **diagnostics.get(key, {}),
            )
        )
    result = dict(methods=reports, comparisons=comparisons)
    write_json(output / split / "records.json", records)
    write_json(output / split / "summary.json", result)
    status(
        output,
        "evaluated",
        split=split,
        EA={k: 100 * v["task_macro_EA"] for k, v in reports.items()},
        unified_element_minus_fixed=comparisons["unified_element_minus_fixed_fusion"],
    )
    return result


def audit_split(split, reference, coefficients, models, out):
    rows, baseline, _, _, pd, sd = inputs(split, reference)
    multi = artifacts.read_json(FEATURES / f"{split}.json")
    single = artifacts.read_json(FEATURES / f"{split}-single.json")
    evidence, native, support = evidence_rows(rows, pd, sd, multi)
    min_data = [restrict_candidates(f, native[f["action_key"]]) for f in multi]
    metric = Metrics(baseline)
    preds, all_records = {}, {}
    for name, folder in [
        ("main", "main"),
        ("single", "single_mlp"),
        ("linear", "linear_20261012"),
    ]:
        records = artifacts.read_json(PREDICTIONS / f"{folder}-{split}.json")
        preds[name] = {r["action_key"]: r["winner"] for r in records}
        all_records[name] = records
    packed = packed_evidence(evidence)
    for name, coeff in [("fixed", (1.0, 1.0, 1.0, 2.0)), ("tuned_fixed", coefficients)]:
        scores = fixed_scores(packed, coeff)
        preds[name] = choose_scores(evidence, scores)
        if name == "fixed":
            for r, f, values in zip(rows, multi, scores):
                assert np.allclose(
                    values[: len(f["candidate_ids"])],
                    f["base_scores"],
                    atol=1e-10,
                    rtol=0,
                )
    replay, _ = predict(models["main"], multi, "cpu")
    assert replay == preds["main"]
    preds["main_restricted"], _ = predict(models["main"], min_data, "cpu")
    for name, model in models.items():
        if name.startswith(("all_", "min_")):
            preds[name], records = predict(
                model, min_data if name.startswith("min_") else multi, "cpu"
            )
            write_json(out / name / split / "records.json", records)
    # Five fixed permutation seeds, within action: perturb only feature 12.
    for seed in range(5):
        rng = np.random.default_rng(20261021 + seed)
        perturbed = []
        for f in multi:
            x = np.array(f["features"], dtype=float)
            if len(x):
                x[:, 11] = x[rng.permutation(len(x)), 11]
            perturbed.append(dict(f, features=x.tolist()))
        preds[f"position_permuted_{seed}"], _ = predict(
            models["main"], perturbed, "cpu"
        )
    correct = {n: correctness(rows, p) for n, p in preds.items()}
    coverage, local, records = {}, {}, []
    for r, f, first in zip(rows, multi, single):
        k = r["action_key"]
        gold = {i for i, v in enumerate(r["labels"]) if v}
        assert first["action_key"] == k
        coverage[k] = coverage_group(first["candidate_ids"], f["candidate_ids"], gold)
        local[k] = local_group(f["candidate_ids"], native[k], gold)
        records.append(
            dict(
                index=r["index"],
                action_key=k,
                task_id=r["task_id"],
                coverage=coverage[k],
                local=local[k],
                positives=sorted(gold),
                hits=f["candidate_ids"],
                first_hits=first["candidate_ids"],
                smallest=sorted(native[k]),
                added_positive_in_B2=coverage[k] == "B2"
                and bool(
                    (set(f["candidate_ids"]) - set(first["candidate_ids"])) & gold
                ),
                winners={n: p[k] for n, p in preds.items()},
                correct={n: v[k] for n, v in correct.items()},
            )
        )
    # Unprepared actions remain failures in C; they contribute exactly zero.
    for k in metric.keys:
        coverage.setdefault(k, "C")
        local.setdefault(k, "unprepared")
    coverage_result = {
        g: metric.difference(
            correct["single"],
            correct["main"],
            {k for k, v in coverage.items() if v == g},
        )
        for g in ("A", "B1", "B2", "C")
    }
    fixed_by_coverage = {
        g: metric.difference(
            correct["fixed"],
            correct["main"],
            {k for k, v in coverage.items() if v == g},
        )
        for g in ("A", "B1", "B2", "C")
    }
    overall = metric.difference(correct["single"], correct["main"])
    for name in ("delta_pp", "fixes", "breaks"):
        assert math.isclose(
            sum(r[name] for r in coverage_result.values()), overall[name], abs_tol=1e-9
        )
    local_result = {}
    for g in ("L", "L_prime", "no_supported_positive", "unprepared"):
        keys = {k for k, v in local.items() if v == g}
        local_result[g] = dict(
            actions=len(keys),
            contribution=metric.difference(
                correct["main_restricted"], correct["main"], keys
            ),
            methods={
                n: dict(
                    correct=sum(v.get(k, False) for k in keys),
                    micro_percent=100 * sum(v.get(k, False) for k in keys) / len(keys)
                    if keys
                    else None,
                )
                for n, v in correct.items()
                if not n.startswith("position_")
            },
            non_minimum_selections={
                n: sum(
                    p.get(k) is not None
                    and p[k] not in native.get(k, set())
                    and p[k] in support.get(k, {})
                    for k in keys
                )
                for n, p in preds.items()
                if not n.startswith("position_")
            },
        )
        local_result[g]["non_minimum_selection_rate_percent"] = {
            n: 100 * v / len(keys) if keys else None
            for n, v in local_result[g]["non_minimum_selections"].items()
        }
    c50_positive = {r["action_key"]: any(r["labels"]) for r in rows}
    hit_positive = {
        r["action_key"]: bool(
            set(f["candidate_ids"]) & {i for i, v in enumerate(r["labels"]) if v}
        )
        for r, f in zip(rows, multi)
    }
    selectable = {
        r["action_key"]: hit_positive[r["action_key"]]
        if f["candidate_ids"]
        else bool(r["labels"][f["fallback"]])
        for r, f in zip(rows, multi)
    }
    # baseline.topk_hit is assigned after image/geometry preparation, so it is
    # insufficient for the pre-preparation recall ceiling on missing inputs.
    if split != "valid":
        prepared_flags = {b["sample_id"]: bool(b["topk_hit"]) for b in baseline}
        assert all(prepared_flags[k] == v for k, v in c50_positive.items())
        full_c50, _ = full_candidate_recall(
            split, baseline, c50_positive, reference, out / split
        )
    else:
        full_c50 = None
    ceilings = dict(
        prepared_C50=metric.report(c50_positive),
        hit_union=metric.report(hit_positive),
        hit_union_plus_actual_no_hit_fallback=metric.report(selectable),
    )
    if full_c50 is not None:
        ceilings["full_C50_recall"] = metric.report(full_c50)
    paired_seeds = []
    for seed in SEEDS:
        a = metric.task_scores(metric.vector(correct[f"all_{seed}"]))
        b = metric.task_scores(metric.vector(correct[f"min_{seed}"]))
        paired_seeds.append((a - b) * 100)
    average_delta = np.mean(paired_seeds, axis=0)
    seed_result = dict(
        all_macro_percent=[
            metric.report(correct[f"all_{s}"])["macro_percent"] for s in SEEDS
        ],
        min_macro_percent=[
            metric.report(correct[f"min_{s}"])["macro_percent"] for s in SEEDS
        ],
        seed_deltas_pp=[float(a.mean()) for a in paired_seeds],
        mean_delta_pp=float(average_delta.mean()),
        ci95_mean_seed_pp=np.quantile(
            average_delta[metric.bootstrap].mean(1), [0.025, 0.975]
        ).tolist(),
    )
    result = dict(
        methods={n: metric.report(v) for n, v in correct.items()},
        ceilings=ceilings,
        main_minus_tuned=metric.difference(correct["tuned_fixed"], correct["main"]),
        main_minus_fixed=metric.difference(correct["fixed"], correct["main"]),
        main_minus_restricted=metric.difference(
            correct["main_restricted"], correct["main"]
        ),
        multi_minus_first=overall,
        coverage=coverage_result,
        main_minus_fixed_by_coverage=fixed_by_coverage,
        B2_actions_adding_positive=sum(r["added_positive_in_B2"] for r in records),
        local=local_result,
        matched_retraining=seed_result,
        candidate_counts=dict(
            all_hit=sum(len(r["candidate_ids"]) for r in multi),
            smallest=sum(len(r["candidate_ids"]) for r in min_data),
        ),
        local_case_indices=[
            r["index"] for r in records if r["local"] == "L" and r["correct"]["main"]
        ],
    )
    write_json(out / split / "records.json", records)
    write_json(out / split / "summary.json", result)
    print(
        json.dumps(
            dict(
                split=split,
                main=result["methods"]["main"]["macro_percent"],
                tuned=result["methods"]["tuned_fixed"]["macro_percent"],
                main_minus_tuned=result["main_minus_tuned"],
                coverage={g: r["delta_pp"] for g, r in coverage_result.items()},
                local_L=local_result["L"]["actions"],
                restricted_delta=result["main_minus_restricted"]["delta_pp"],
                retrained=seed_result,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def evaluate_mechanisms(out, reference, started):
    results = {
        split: analyze(split, out, reference)
        for split in ("valid", *reference["splits"])
    }
    write_json(
        out / "summary.json",
        dict(splits=results, seconds=time.monotonic() - started),
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
        for split, result in results.items():
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
    status(out, "complete", seconds=time.monotonic() - started)


def evaluate_functional(out, reference, coeff, models, training, started):
    results = {
        split: audit_split(split, reference, coeff, models, out)
        for split in ("valid", *reference["splits"])
    }
    write_json(
        out / "summary.json",
        dict(
            status="complete",
            coefficients=coeff,
            training=training,
            splits=results,
            seconds=time.monotonic() - started,
        ),
    )
    write_json(
        out / "verification.json",
        dict(
            status="passed",
            main_prediction_replay=True,
            original_fixed_score_replay=True,
            coverage_contributions_add_to_total=True,
            prepared_C50_matches_frozen_topk_hit=True,
            actions=sum(r["methods"]["main"]["actions"] for r in results.values()),
        ),
    )
    print("Completed in", time.monotonic() - started, "seconds", flush=True)
