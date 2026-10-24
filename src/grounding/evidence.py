"""Candidate geometry and fixed evidence aggregation; no files or model loading."""

import math

import numpy as np


def containing_candidates(row, point):
    if point is None:
        return []
    x, y = point
    return [
        i
        for i, box in enumerate(row["candidate_bboxes"])
        if box is not None
        and box[2] > 0
        and box[3] > 0
        and box[0] <= x < box[0] + box[2]
        and box[1] <= y < box[1] + box[3]
    ]


def log_sigmoid(value):
    return min(value, 0.0) - math.log1p(math.exp(-abs(value)))


def logsumexp_values(values):
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def row_weights(row):
    """Candidate-count prior from the row's own windows (C50 geometry)."""
    counts = [len(window["centres"]) for window in row["windows"]]
    total = sum(counts) or 1
    return [math.log(count) - math.log(total) for count in counts]


def packed_evidence(evidence):
    width = max(1, max(len(r["ids"]) for r in evidence))
    shape = (len(evidence), width, 3)
    lp, prior, present = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=bool)
    sem, tail = np.zeros(shape[:2]), np.zeros(shape[:2])
    for a, r in enumerate(evidence):
        for j, i in enumerate(r["ids"]):
            n = len(r["logps"][j])
            lp[a, j, :n] = r["logps"][j]
            prior[a, j, :n] = r["priors"][j]
            present[a, j, :n] = True
            sem[a, j] = r["semantics"][j]
            tail[a, j] = i >= 20
    return lp, prior, present, sem, tail


def fixed_scores(packed, coefficients):
    lp, prior, present, sem, tail = packed
    a, b, c, d = coefficients
    values = np.where(present, a * lp + b * prior, -np.inf)
    return np.logaddexp.reduce(values, axis=2) + c * sem - d * tail


def choose_scores(evidence, scores):
    return {
        r["key"]: r["ids"][int(np.argmax(s[: len(r["ids"])]))]
        if r["ids"]
        else r["fallback"]
        for r, s in zip(evidence, scores)
    }


def document_point(point, crop_size, top):
    """Invert the exact 1000-unit crop coordinate convention; never clamp."""
    return [point[0] * crop_size[0] / 1000, top + point[1] * crop_size[1] / 1000]


def proposal_info(row, points):
    boxes = row["candidate_bboxes"]
    priors = row_weights(row)
    hits, existing = {}, set()
    for wi, (window, prior) in enumerate(zip(points["windows"], priors)):
        point = (
            document_point(window["coordinate"], window["crop_size"], window["top"])
            if window["coordinate"] is not None
            else None
        )
        containing = containing_candidates(row, point)
        if containing:
            existing.add(min(containing, key=lambda i: (boxes[i][2] * boxes[i][3], i)))
        for i in containing:
            hits.setdefault(i, []).append(
                dict(
                    window=wi,
                    point=point,
                    prediction=window,
                    evidence=window["log_probability"]
                    + prior
                    - (2.0 if i >= 20 else 0.0),
                )
            )
    challengers = sorted(hits.keys() - existing)
    return hits, existing, challengers
