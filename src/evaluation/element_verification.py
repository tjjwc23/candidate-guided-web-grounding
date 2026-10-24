"""evaluation / element verification."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import artifacts as artifacts
import manifest as config
from artifacts import write_json
from data.element_inputs import inputs
from grounding.element_scoring import load_model
from grounding.element_scoring import predict as predict


def manual_report(rows, baseline, records):
    labels = {r["action_key"]: r["labels"] for r in rows}
    pred = {r["action_key"]: r["winner"] for r in records}
    assert len(pred) == len(records) == len(rows)
    assert set(pred) == set(labels)
    correct = {k: bool(labels[k][v]) for k, v in pred.items()}
    counts = Counter(r["annotation_id"] for r in baseline)
    totals = defaultdict(int)
    for r in baseline:
        totals[r["annotation_id"]] += correct.get(r["sample_id"], False)
    score = sum(totals[t] / n for t, n in counts.items()) / len(counts)
    return score, correct, len(counts), sum(counts.values())


def manual_pair(baseline, first, second):
    tasks = defaultdict(list)
    for r in baseline:
        key = r["sample_id"]
        tasks[r["annotation_id"]].append(
            int(second.get(key, False)) - int(first.get(key, False))
        )
    gains = np.array([np.mean(tasks[t]) for t in sorted(tasks)])
    rng = np.random.default_rng(26092026)
    sampled = gains[rng.integers(len(gains), size=(10000, len(gains)))].mean(1) * 100
    return dict(
        delta_pp=float(gains.mean() * 100),
        task_bootstrap95_pp=np.quantile(sampled, [0.025, 0.975]).tolist(),
        fixes=sum(v == 1 for a in tasks.values() for v in a),
        breaks=sum(v == -1 for a in tasks.values() for v in a),
    )


def controls_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=str, required=True)
    parser.add_argument("--stability", type=str, required=True)
    parser.add_argument("--linear-models", type=Path, required=True,
                        help="Native exports with linear_<seed>/head.pt for each compared seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controls, stability = Path(args.controls), Path(args.stability)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source = artifacts.read_json(controls / "summary.json")
    seeds = artifacts.read_json(stability / "summary.json")
    reference = config.load()
    results = {}
    verified = {}
    for split in ("valid", *reference["splits"]):
        rows, baseline, features, _, _, _ = inputs(split, reference)
        correct = {}
        for name, expected in source["splits"][split]["methods"].items():
            metric, correct[name], tasks, actions = manual_report(
                rows,
                baseline,
                artifacts.read_json(controls / split / f"{name}_records.json"),
            )
            assert abs(metric - expected["task_macro_EA"]) < 1e-12
            assert sum(correct[name].values()) == expected["correct_steps"]
            assert (
                tasks == expected["tasks"] and actions == expected["denominator_steps"]
            )
        for name in ("single_mlp", "multi_linear"):
            actual = manual_pair(baseline, correct[name], correct["multi_mlp"])
            expected = source["splits"][split]["comparisons"][f"multi_mlp_minus_{name}"]
            for key in ("delta_pp", "task_bootstrap95_pp", "fixes", "breaks"):
                assert np.allclose(actual[key], expected[key], atol=1e-10, rtol=0)
        runs = {}
        for seed in (20261012, 20261013, 20261014):
            folder = args.linear_models / f"linear_{seed}"
            linear = load_model(folder / "head.pt")
            _, records = predict(linear.eval(), features, "cpu")
            write_json(output / f"linear_{seed}" / split / "records.json", records)
            linear_metric, linear_correct, _, _ = manual_report(rows, baseline, records)
            mlp_metric, mlp_correct, _, _ = manual_report(
                rows,
                baseline,
                artifacts.read_json(stability / str(seed) / split / "records.json"),
            )
            assert (
                abs(
                    mlp_metric
                    - seeds["splits"][split]["methods"][str(seed)]["task_macro_EA"]
                )
                < 1e-12
            )
            runs[seed] = dict(
                mlp_EA_percent=100 * mlp_metric,
                linear_EA_percent=100 * linear_metric,
                mlp_minus_linear=manual_pair(baseline, linear_correct, mlp_correct),
            )
        result = dict(runs=runs)
        for key in ("mlp", "linear"):
            values = [v[f"{key}_EA_percent"] for v in runs.values()]
            result[key] = dict(
                mean_EA_percent=statistics.mean(values),
                sample_sd_pp=statistics.stdev(values),
            )
        differences = [v["mlp_minus_linear"]["delta_pp"] for v in runs.values()]
        result["paired_seed_delta_mean_pp"] = statistics.mean(differences)
        result["paired_seed_delta_sample_sd_pp"] = statistics.stdev(differences)
        results[split] = result
        verified[split] = dict(
            prepared_actions=len(rows),
            full_actions=len(baseline),
            original_metrics_and_intervals=True,
            seed_metrics=True,
        )
    write_json(
        output / "matched_summary.json",
        dict(
            status="complete",
            seeds=[20261012, 20261013, 20261014],
            device="cpu",
            splits=results,
            scope="Descriptive mean and sample SD of 3 validation-selected seeds; original GPU checkpoint remains separate",
            seed_comparison=seeds["splits"],
            implementation_sha256=artifacts.file_digest(Path(__file__)),
        ),
    )
    write_json(
        output / "independent_verification.json",
        dict(
            status="passed",
            splits=verified,
            method="Independent task counting, correctness aggregation, task-macro deltas and bootstrap from saved winners; no use of element_report or experiments.paired",
            implementation_sha256=artifacts.file_digest(Path(__file__)),
        ),
    )
    print(
        "Independent metrics/interval replay passed; matched three-seed summary written.",
        flush=True,
    )
    for split, r in results.items():
        print(
            split,
            {k: r[k] for k in ("mlp", "linear", "paired_seed_delta_mean_pp")},
            flush=True,
        )


def inside(box, point):
    return (
        box is not None
        and point is not None
        and box[2] > 0
        and box[3] > 0
        and (
            box[0] <= point[0] < box[0] + box[2]
            and box[1] <= point[1] < box[1] + box[3]
        )
    )


def functional_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    summary = artifacts.read_json(out / "summary.json")
    coeff = summary["coefficients"]
    checks = {}
    tuning = artifacts.read_json(out / "tuning.json")
    best = max(tuning["trials"], key=lambda r: r["validation_macro_percent"])
    assert best["coefficients"] == list(coeff)
    for split, result in summary["splits"].items():
        rows, baseline, _, _, pd, sd = inputs(split, config.load())
        records = artifacts.read_json(out / split / "records.json")
        by_key = {r["action_key"]: r for r in records}
        counts = Counter(b["annotation_id"] for b in baseline)
        totals = Counter()
        macro = Counter()
        groups = Counter()
        fixed_groups = Counter()
        local_counts = Counter()
        for row in rows:
            k = row["action_key"]
            r = by_key[k]
            boxes = row["candidate_bboxes"]
            points = artifacts.read_json(pd / f"{row['index']}.json")["windows"]
            sem = artifacts.read_json(sd / f"{row['index']}.json")
            logits = sem["semantic_logits"]
            n = [len(w["centres"]) for w in row["windows"]]
            by_window = []
            for w in points:
                z = w["coordinate"]
                p = (
                    None
                    if z is None
                    else [
                        z[0] * w["crop_size"][0] / 1000,
                        w["top"] + z[1] * w["crop_size"][1] / 1000,
                    ]
                )
                by_window.append({i for i, b in enumerate(boxes) if inside(b, p)})
            hit = set().union(*by_window)
            first = by_window[0]
            native = {
                min(ids, key=lambda i: (boxes[i][2] * boxes[i][3], i))
                for ids in by_window
                if ids
            }
            gold = {i for i, v in enumerate(row["labels"]) if v}
            assert (
                sorted(hit) == r["hits"]
                and sorted(native) == r["smallest"]
                and sorted(first) == r["first_hits"]
            )
            g = (
                "C"
                if not (gold & hit)
                else ("A" if not (gold & first) else ("B1" if first == hit else "B2"))
            )
            assert g == r["coverage"]
            l = (
                "no_supported_positive"
                if not (gold & hit)
                else ("L_prime" if gold & native else "L")
            )
            assert l == r["local"]
            local_counts[l] += 1
            weight = 1 / counts[row["task_id"]] / len(counts)
            for name, i in r["winners"].items():
                ok = i is not None and i in gold
                assert ok == r["correct"][name]
                totals[name] += ok
                macro[name] += weight * ok
                if name.startswith("min_") or name == "main_restricted":
                    assert (
                        i in native
                        if native
                        else i == max(range(len(logits)), key=lambda j: (logits[j], -j))
                    )
            for name, c in [("fixed", (1, 1, 1, 2)), ("tuned_fixed", coeff)]:
                values = {}
                for i in hit:
                    support = [
                        c[0] * points[w]["log_probability"]
                        + c[1] * math.log(n[w] / sum(n))
                        for w, ids in enumerate(by_window)
                        if i in ids
                    ]
                    hi = max(support)
                    lp = hi + math.log(sum(math.exp(v - hi) for v in support))
                    ls = min(logits[i], 0) - math.log1p(math.exp(-abs(logits[i])))
                    values[i] = lp + c[2] * ls - c[3] * (i >= 20)
                winner = (
                    max(values, key=lambda i: (values[i], -i))
                    if values
                    else max(range(len(logits)), key=lambda j: (logits[j], -j))
                )
                assert winner == r["winners"][name], (split, row["index"], name)
            groups[g] += (
                100 * weight * (int(r["correct"]["main"]) - int(r["correct"]["single"]))
            )
            fixed_groups[g] += (
                100 * weight * (int(r["correct"]["main"]) - int(r["correct"]["fixed"]))
            )
        for name, r in result["methods"].items():
            assert totals[name] == r["correct"]
            assert abs(macro[name] * 100 - r["macro_percent"]) < 1e-10
            assert abs(totals[name] / len(baseline) * 100 - r["micro_percent"]) < 1e-10
        for g, r in result["coverage"].items():
            assert abs(groups[g] - r["delta_pp"]) < 1e-10
        for g, r in result["main_minus_fixed_by_coverage"].items():
            assert abs(fixed_groups[g] - r["delta_pp"]) < 1e-10
        for g, n in local_counts.items():
            assert result["local"][g]["actions"] == n
        assert (
            result["ceilings"]["hit_union_plus_actual_no_hit_fallback"]["macro_percent"]
            + 1e-10
            >= result["methods"]["main"]["macro_percent"]
        )
        # Independently construct averaged per-task differences and bootstrap.
        tasks = sorted(counts)
        ti = {t: i for i, t in enumerate(tasks)}
        deltas = []
        for seed in (20261012, 20261013, 20261014):
            v = np.zeros(len(tasks))
            for b in baseline:
                r = by_key.get(b["sample_id"])
                if r:
                    v[ti[b["annotation_id"]]] += (
                        (
                            int(r["correct"][f"all_{seed}"])
                            - int(r["correct"][f"min_{seed}"])
                        )
                        / counts[b["annotation_id"]]
                        * 100
                    )
            deltas.append(v)
        avg = np.mean(deltas, axis=0)
        idx = np.random.default_rng(26092026).integers(
            len(tasks), size=(10000, len(tasks)), dtype=np.int32
        )
        ci = np.quantile(avg[idx].mean(1), [0.025, 0.975])
        assert np.allclose(
            ci, result["matched_retraining"]["ci95_mean_seed_pp"], atol=1e-10, rtol=0
        )
        checks[split] = dict(
            actions=len(rows),
            denominator=len(baseline),
            geometry=True,
            fixed_and_tuned_decisions=True,
            metrics=True,
            coverage=True,
            local=True,
            paired_seed_interval=True,
        )
    write_json(
        out / "independent_verification.json",
        dict(
            status="passed",
            splits=checks,
            verification_script_sha256=artifacts.file_digest(Path(__file__)),
            limitation="Geometry, scores and reports replayed from cached inputs; raw model generation and new unseen-site generalization not tested",
        ),
    )
    print("Independent functional audit replay passed", flush=True)
