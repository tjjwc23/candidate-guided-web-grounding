"""Pure full-denominator Grounding metrics and paired task statistics."""

from collections import Counter, defaultdict

import numpy as np


def coverage_group(first, multi, positives):
    first, multi, positives = set(first), set(multi), set(positives)
    assert first <= multi
    if not multi & positives:
        return "C"
    if not first & positives:
        return "A"
    return "B1" if first == multi else "B2"


def local_group(hit, smallest, positives):
    hit, smallest, positives = set(hit), set(smallest), set(positives)
    assert smallest <= hit
    if not hit & positives:
        return "no_supported_positive"
    return "L_prime" if smallest & positives else "L"


class Metrics:
    def __init__(self, baseline):
        self.baseline = baseline
        self.keys = [b["sample_id"] for b in baseline]
        counts = Counter(b["annotation_id"] for b in baseline)
        tasks = sorted(counts)
        ti = {t: i for i, t in enumerate(tasks)}
        self.task_ids = np.array([ti[b["annotation_id"]] for b in baseline])
        self.within_weight = np.array(
            [1 / counts[b["annotation_id"]] for b in baseline]
        )
        self.tasks = len(tasks)
        self.bootstrap = np.random.default_rng(26092026).integers(
            len(tasks), size=(10000, len(tasks)), dtype=np.int32
        )

    def vector(self, correct):
        return np.array([float(correct.get(k, False)) for k in self.keys])

    def task_scores(self, values):
        return np.bincount(
            self.task_ids, weights=values * self.within_weight, minlength=self.tasks
        )

    def report(self, correct):
        x = self.vector(correct)
        return dict(
            correct=int(x.sum()),
            actions=len(x),
            tasks=self.tasks,
            micro_percent=float(x.mean() * 100),
            macro_percent=float(self.task_scores(x).mean() * 100),
        )

    def difference(self, first, second, keys=None):
        a, b = self.vector(first), self.vector(second)
        mask = (
            np.ones(len(a), dtype=bool)
            if keys is None
            else np.array([k in keys for k in self.keys])
        )
        delta = self.task_scores((b - a) * mask)
        return dict(
            actions=int(mask.sum()),
            delta_pp=float(delta.mean() * 100),
            ci95_pp=np.quantile(
                delta[self.bootstrap].mean(1) * 100, [0.025, 0.975]
            ).tolist(),
            fixes=int(((b > a) & mask).sum()),
            breaks=int(((b < a) & mask).sum()),
            first_micro_percent=float(a[mask].mean() * 100) if mask.any() else None,
            second_micro_percent=float(b[mask].mean() * 100) if mask.any() else None,
        )


def correctness(rows, predictions):
    return {
        r["action_key"]: predictions.get(r["action_key"]) is not None
        and bool(r["labels"][predictions[r["action_key"]]])
        for r in rows
    }


def evaluate_validation(predictions, rows, labels, baseline):
    task_correct = Counter()
    task_total = Counter(r["annotation_id"] for r in baseline)
    for row in rows:
        winner = predictions[row["action_key"]]
        task_correct[row["task_id"]] += winner is not None and bool(
            labels[row["action_key"]][winner]
        )
    return sum(task_correct[t] / n for t, n in task_total.items()) / len(task_total)


def paired(baseline, first, second):
    counts = Counter(r["annotation_id"] for r in baseline)
    gains = defaultdict(float)
    fixes = breaks = 0
    for row in baseline:
        key, task = row["sample_id"], row["annotation_id"]
        a, b = bool(first.get(key, False)), bool(second.get(key, False))
        fixes += b and not a
        breaks += a and not b
        gains[task] += (int(b) - int(a)) / counts[task]
    values = np.array([gains[t] for t in sorted(counts)])
    rng = np.random.default_rng(26092026)
    boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(1) * 100
    return dict(
        delta_pp=float(values.mean() * 100),
        fixes=int(fixes),
        breaks=int(breaks),
        task_bootstrap95_pp=np.quantile(boot, [0.025, 0.975]).tolist(),
    )


def task_macro_weights(baseline):
    """Return per-action macro weights, per-task action counts and task count."""
    per_task = Counter(row["annotation_id"] for row in baseline)
    tasks = len(per_task)
    weight = {
        row["sample_id"]: 1.0 / (per_task[row["annotation_id"]] * tasks)
        for row in baseline
    }
    return weight, per_task, tasks


def element_report(rows, decisions, baseline):
    """Task-macro element accuracy on the full denominator plus the eligible subset."""
    weight, per_task, tasks = task_macro_weights(baseline)
    correct = {}
    for row in rows:
        winner = decisions.get(row["action_key"])
        if winner is not None:
            correct[row["action_key"]] = bool(row["labels"][winner])
    macro = sum(weight.get(key, 0.0) for key, value in correct.items() if value)
    eligible = [row for row in rows if any(row["labels"])]
    eligible_correct = sum(1 for row in eligible if correct.get(row["action_key"]))
    return {
        "steps_scored": len(rows),
        "tasks": tasks,
        "denominator_steps": sum(per_task.values()),
        "correct_steps": sum(correct.values()),
        "micro_EA": sum(correct.values()) / sum(per_task.values()),
        "task_macro_EA": macro,
        "eligible_steps": len(eligible),
        "eligible_correct": eligible_correct,
        "eligible_micro_EA": eligible_correct / len(eligible) if eligible else None,
        "steps_without_decision": sum(
            1 for row in rows if decisions.get(row["action_key"]) is None
        ),
    }
