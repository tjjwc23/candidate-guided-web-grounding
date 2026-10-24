"""grounding / element scoring."""

from __future__ import annotations

import torch
from torch import nn

from pathlib import Path
from model_artifacts import seal_export, read_export
from grounding.element_features import FEATURE_NAMES


def collate(rows, device, labels=None, task_counts=None):
    width = max(1, max(len(row["candidate_ids"]) for row in rows))
    batch = dict(
        features=torch.zeros(len(rows), width, len(FEATURE_NAMES), device=device),
        base_scores=torch.zeros(len(rows), width, dtype=torch.float64, device=device),
        valid=torch.zeros(len(rows), width, dtype=torch.bool, device=device),
    )
    if labels is not None:
        batch["positive"] = torch.zeros(
            len(rows), width, dtype=torch.bool, device=device
        )
    for bi, row in enumerate(rows):
        n = len(row["candidate_ids"])
        if not n:
            continue
        batch["features"][bi, :n] = torch.tensor(row["features"], device=device)
        batch["base_scores"][bi, :n] = torch.tensor(
            row["base_scores"], dtype=torch.float64, device=device
        )
        batch["valid"][bi, :n] = True
        if labels is not None:
            batch["positive"][bi, :n] = torch.tensor(
                [bool(labels[row["action_key"]][i]) for i in row["candidate_ids"]],
                device=device,
            )
    if task_counts is not None:
        batch["task_weight"] = torch.tensor(
            [1.0 / task_counts[row["task_id"]] for row in rows], device=device
        )
    return batch


def ranking_loss(model, batch):
    assert batch["positive"].any(1).all(), (
        "Ranking gradients require a supported positive"
    )
    scores = model(batch)
    losses = scores.logsumexp(1) - scores.masked_fill(
        ~batch["positive"], -1e9
    ).logsumexp(1)
    return (losses * batch["task_weight"]).sum() / batch["task_weight"].sum()


@torch.no_grad()
def predict(model, rows, device):
    predictions, records = {}, []
    for start in range(0, len(rows), 128):
        chunk = rows[start : start + 128]
        scores = model(collate(chunk, device)).cpu().tolist()
        for row, values in zip(chunk, scores):
            ids = row["candidate_ids"]
            position = (
                max(range(len(ids)), key=lambda j: (values[j], -ids[j]))
                if ids
                else None
            )
            winner = ids[position] if ids else row["fallback"]
            predictions[row["action_key"]] = winner
            records.append(
                dict(
                    index=row["index"],
                    action_key=row["action_key"],
                    winner=winner,
                    candidate_ids=ids,
                    semantic_recovery=not bool(ids),
                    candidate_log_scores={str(i): values[j] for j, i in enumerate(ids)},
                )
            )
    return predictions, records


class ElementScorer(nn.Module):
    """Shared residual scorer; layout and initialization are checkpoint contracts."""

    def __init__(self, *, linear=False):
        super().__init__()
        feature_count = len(FEATURE_NAMES)
        if linear:
            self.correction = nn.Linear(feature_count, 1, bias=False)
            nn.init.zeros_(self.correction.weight)
        else:
            self.correction = nn.Sequential(
                nn.Linear(feature_count, 40),
                nn.ReLU(),
                nn.Linear(40, 40),
                nn.ReLU(),
                nn.Linear(40, 1, bias=False),
            )
            nn.init.zeros_(self.correction[-1].weight)

    def forward(self, batch):
        residual = self.correction(batch["features"]).squeeze(-1).double()
        return (batch["base_scores"] + residual).masked_fill(~batch["valid"], -1e9)


class LinearScorer(ElementScorer):
    def __init__(self):
        super().__init__(linear=True)


class SubsetReranker(ElementScorer):
    def __init__(self, keep):
        super().__init__()
        self.keep = list(keep)
        old = self.correction[0]
        new = nn.Linear(len(keep), 40)
        with torch.no_grad():
            new.weight.copy_(old.weight[:, self.keep])
            new.bias.copy_(old.bias)
        self.correction[0] = new

    def forward(self, batch):
        return super().forward(dict(batch, features=batch["features"][:, :, self.keep]))


def save_model(model, path):
    """Export native element weights with their explicit architecture."""
    path = Path(path)
    torch.save(model.state_dict(), path)
    seal_export(path.parent, kind="element", contract="element18-v1", files=[path.name],
                metadata=dict(weights=path.name, feature_names=list(FEATURE_NAMES),
                              architecture="linear" if isinstance(model, LinearScorer) else "mlp",
                              feature_indices=getattr(model, "keep", list(range(18)))))


def load_model(path):
    """Load only a sealed native export; old dimensions and raw checkpoints fail."""
    path = Path(path)
    value = read_export(path.parent, kind="element", contract="element18-v1")
    meta = value["metadata"]
    if meta["weights"] != path.name or path.name not in value["files"] or meta["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Element export layout changed")
    keep = meta["feature_indices"]
    if not keep or len(set(keep)) != len(keep) or any(i not in range(18) for i in keep):
        raise ValueError("Invalid native element feature indices")
    if meta["architecture"] == "linear":
        if keep != list(range(18)):
            raise ValueError("Linear element export requires all 18 features")
        model = LinearScorer()
    elif meta["architecture"] == "mlp":
        model = ElementScorer() if keep == list(range(18)) else SubsetReranker(keep)
    else:
        raise ValueError("Unknown element architecture")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
