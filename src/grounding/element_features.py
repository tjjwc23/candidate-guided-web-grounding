"""Label-free 18D element evidence for full-width windows."""

import math
from collections import Counter

from grounding.evidence import log_sigmoid, logsumexp_values
from grounding.evidence import proposal_info

FEATURE_NAMES = [
    "semantic_logit",
    "semantic_logweight",
    "proposal_rank",
    "proposal_tail",
    "candidate_log_evidence",
    "hit_window_fraction",
    "smallest_hit_fraction",
    "coordinate_logp",
    "coordinate_mean_logp",
    "coordinate_tokens",
    "window_logprior",
    "window_document_top",
    "crop_page_height",
    "bbox_crop_width",
    "bbox_crop_height",
    "bbox_crop_logarea",
    "point_bbox_edge_margin",
    "point_bbox_center_distance",
]


def features(row, points, logits):
    """Use label-free evidence; smallest-box membership is a feature, never a filter."""
    assert points["action_key"] == row["action_key"]
    assert len(logits) == len(row["candidate_bboxes"]) <= 50
    assert len(points["windows"]) == len(row["windows"])
    assert all(p["top"] == w["top"] for p, w in zip(points["windows"], row["windows"]))
    hits, _, _ = proposal_info(row, points)
    ids = sorted(hits)
    boxes = row["candidate_bboxes"]
    by_window = {}
    for i, support in hits.items():
        for hit in support:
            by_window.setdefault(hit["window"], []).append(i)
    native_counts = Counter(
        min(choices, key=lambda i: (boxes[i][2] * boxes[i][3], i))
        for choices in by_window.values()
    )
    counts = [len(w["centres"]) for w in row["windows"]]
    priors = [math.log(n / sum(counts)) for n in counts]
    pw, ph = row["page_size"]
    assert all(
        p["crop_size"][0] == pw for p in points["windows"]
    ), "Full-width windows required"
    matrix, base_scores = [], []
    for i in ids:
        support = hits[i]
        evidence = logsumexp_values([v["evidence"] for v in support])
        x, y, w, h = boxes[i]
        observations, weights = [], []
        for hit in support:
            p = hit["prediction"]
            cw, ch = p["crop_size"]
            px, py = hit["point"]
            tokens = max(1, len(p["token_ids"]))
            observations.append(
                [
                    p["log_probability"] / 10,
                    p["log_probability"] / tokens,
                    tokens / 24,
                    priors[hit["window"]],
                    p["top"] / ph,
                    ch / ph,
                    w / cw,
                    h / ch,
                    math.log(w * h / (cw * ch)) / 4,
                    min(
                        (px - x) / cw,
                        (x + w - px) / cw,
                        (py - y) / ch,
                        (y + h - py) / ch,
                    ),
                    ((px - x - w / 2) / w) ** 2 + ((py - y - h / 2) / h) ** 2,
                ]
            )
            weights.append(math.exp(hit["evidence"] - evidence))
        average = [
            sum(weight * v[j] for weight, v in zip(weights, observations))
            for j in range(len(observations[0]))
        ]
        matrix.append(
            [
                logits[i] / 10,
                log_sigmoid(logits[i]) / 4,
                i / 50,
                float(i >= 20),
                evidence / 10,
                len(support) / len(points["windows"]),
                native_counts[i] / len(points["windows"]),
                *average,
            ]
        )
        base_scores.append(evidence + log_sigmoid(logits[i]))
    assert all(
        len(v) == len(FEATURE_NAMES) and all(math.isfinite(x) for x in v)
        for v in matrix
    )
    return dict(
        index=row["index"],
        action_key=row["action_key"],
        task_id=row["task_id"],
        partition=row.get("partition", "test"),
        candidate_ids=ids,
        candidate_count=len(boxes),
        features=matrix,
        base_scores=base_scores,
        fallback=max(range(len(logits)), key=lambda i: (logits[i], -i)),
    )


def first_window_features(row, points, logits):
    return features(
        dict(row, windows=row["windows"][:1]),
        dict(points, windows=points["windows"][:1]),
        logits,
    )


def restrict_candidates(row, retained):
    """Change choice set only; preserve evidence, features, fallback and ordering."""
    positions = [j for j, i in enumerate(row["candidate_ids"]) if i in retained]
    return dict(
        row,
        candidate_ids=[row["candidate_ids"][j] for j in positions],
        features=[row["features"][j] for j in positions],
        base_scores=[row["base_scores"][j] for j in positions],
    )
