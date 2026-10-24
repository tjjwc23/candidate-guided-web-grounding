from __future__ import annotations

import itertools
from collections import Counter

import torch

import artifacts as artifacts
import grounding.element_scoring as element_scoring
from artifacts import write_json
from data.element_inputs import (
    TRAIN,
)
from evaluation.grounding_metrics import (
    task_macro_weights,
    evaluate_validation,
)
from execution import experiment_status as status
from grounding.element_features import FEATURE_NAMES as FEATURE_NAMES
from grounding.element_features import features as features
from grounding.element_scoring import ElementScorer
from grounding.element_scoring import predict as predict
from grounding.evidence import choose_scores, fixed_scores, packed_evidence
from grounding.head_training import fit_head
from manifest import load_recipe

RECIPE = load_recipe("element")
EPOCHS = RECIPE["epochs"]
LEARNING_RATES = tuple(RECIPE["learning_rates"])
SEED = RECIPE["seed"]
GRID = tuple(RECIPE["coefficient_grid"])


def fit_ranking_head(output, dataset, labels, model_factory, seed, metric, **options):
    """18D element recipe: informative rows and task-balanced batches."""
    fit = [r for r in dataset if r["partition"] == "fit"]
    valid = [r for r in dataset if r["partition"] == "valid"]
    informative = [
        r
        for r in fit
        if len(r["candidate_ids"]) > 1
        and any(labels[r["action_key"]][i] for i in r["candidate_ids"])
    ]
    counts = Counter(r["task_id"] for r in fit)
    for key in (
        "learning_rates",
        "epochs",
        "batch_size",
        "weight_decay",
        "gradient_clip",
    ):
        options.setdefault(key, RECIPE[key])
    model, result = fit_head(
        output,
        informative,
        valid,
        model_factory,
        seed,
        metric,
        collate=lambda rows, device: element_scoring.collate(
            rows, device, labels, counts
        ),
        loss_fn=element_scoring.ranking_loss,
        predict=element_scoring.predict,
        **options,
    )
    element_scoring.save_model(model, output / "head.pt")
    result.update(
        fit_actions=len(fit),
        valid_actions=len(valid),
        informative_fit_actions=len(informative),
    )
    return model, result


def train_head(output, dataset, model_type, seed):
    """Native 18D CPU recipe for linear, seed and feature-subset controls."""
    output.mkdir(parents=True, exist_ok=False)
    labels = artifacts.read_json(TRAIN / "supervision.json")
    fit_tasks = {r["task_id"] for r in dataset if r["partition"] == "fit"}
    valid_tasks = {r["task_id"] for r in dataset if r["partition"] == "valid"}
    assert not valid_tasks & fit_tasks
    baseline = [
        r
        for r in artifacts.read_json(TRAIN / "fit_valid_baseline.json")
        if r["annotation_id"] in valid_tasks
    ]
    weights, _, _ = task_macro_weights(baseline=baseline)

    def metric(predictions, rows):
        return sum(weights[k] for k, winner in predictions.items() if labels[k][winner])

    def progress(lr, epoch, value, selected, elapsed):
        if epoch == EPOCHS:
            status(output, "training", lr=lr, selected=selected, seconds=elapsed)

    model, result = fit_ranking_head(
        output,
        dataset,
        labels,
        model_type,
        seed,
        metric,
        budget_seconds=120,
        on_epoch=progress,
    )
    write_json(output / "validation_records.json", result["validation_records"])
    summary = dict(
        selected=result["selected"],
        seed=seed,
        device="cpu",
        parameters=sum(p.numel() for p in model.parameters()),
        informative_fit_actions=result["informative_fit_actions"],
        fit_actions=result["fit_actions"],
        valid_actions=result["valid_actions"],
        seconds=result["seconds"],
        head_sha256=artifacts.file_digest(output / "head.pt"),
        runs=result["runs"],
    )
    write_json(output / "training_summary.json", summary)
    return model, summary


def fit_native_head(dataset, labels, baseline, seed, out):
    """Native 18D CPU recipe used by matched candidate-policy controls."""
    from grounding.element_scoring import ElementScorer

    out.mkdir(parents=True, exist_ok=False)
    weights, _, _ = task_macro_weights(baseline=baseline)

    def metric(predictions, rows):
        return sum(weights[k] for k, i in predictions.items() if labels[k][i])

    model, result = fit_ranking_head(
        out, dataset, labels, ElementScorer, seed, metric, budget_seconds=120
    )
    history = [
        dict(
            lr=run["lr"],
            epoch=row["epoch"],
            validation_macro_percent=row["valid_task_macro_EA"] * 100,
        )
        for run in result["runs"]
        for row in run["history"]
    ]
    selected = result["selected"]
    summary = dict(
        seed=seed,
        selected=dict(
            lr=selected["lr"],
            epoch=selected["epoch"],
            validation_macro_percent=selected["valid_task_macro_EA"] * 100,
        ),
        informative_fit=result["informative_fit_actions"],
        seconds=result["seconds"],
        history=history,
    )
    write_json(out / "training_summary.json", summary)
    return model, summary


def train(output, dataset, source, *, device="cuda"):
    labels = artifacts.read_json(source / "supervision.json")
    valid_tasks = {r["task_id"] for r in dataset if r["partition"] == "valid"}
    baseline = [
        r
        for r in artifacts.read_json(source / "fit_valid_baseline.json")
        if r["annotation_id"] in valid_tasks
    ]

    def metric(predictions, rows):
        return evaluate_validation(predictions, rows, labels, baseline)

    def progress(lr, epoch, value, selected, elapsed):
        if epoch % 5 == 0:
            status(
                output,
                "training",
                lr=lr,
                epoch=epoch,
                valid_EA=value,
                selected=selected,
            )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _, result = fit_ranking_head(
        output,
        dataset,
        labels,
        ElementScorer,
        SEED,
        metric,
        device=device,
        learning_rates=LEARNING_RATES,
        epochs=EPOCHS,
        on_epoch=progress,
    )
    write_json(output / "validation_records.json", result["validation_records"])
    summary = dict(
        selected=result["selected"],
        informative_fit_actions=result["informative_fit_actions"],
        fit_actions=result["fit_actions"],
        valid_actions=result["valid_actions"],
        seconds=result["seconds"],
        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20
        if device == "cuda"
        else None,
        feature_schema="element18-v1",
        head_sha256=artifacts.file_digest(output / "head.pt"),
        runs=result["runs"],
    )
    write_json(output / "training_summary.json", summary)
    return summary


def tune(valid_rows, baseline, evidence, out):
    packed = packed_evidence(evidence)
    weights, _, _ = task_macro_weights(baseline=baseline)
    by_key = {r["action_key"]: r for r in valid_rows}
    original = (1.0, 1.0, 1.0, 2.0)
    candidates = [original] + [
        x for x in itertools.product(GRID, repeat=4) if x != original
    ]
    history, best, selected = [], -1.0, None
    for coeff in candidates:
        pred = choose_scores(evidence, fixed_scores(packed, coeff))
        value = sum(weights[k] for k, i in pred.items() if by_key[k]["labels"][i])
        history.append(dict(coefficients=coeff, validation_macro_percent=value * 100))
        if value > best + 1e-12:
            best, selected = value, coeff
    write_json(
        out / "tuning.json",
        dict(selected=selected, validation_macro_percent=best * 100, trials=history),
    )
    return selected
