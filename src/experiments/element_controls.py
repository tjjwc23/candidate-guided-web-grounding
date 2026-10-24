"""Orchestrate matched element control fits and their evaluations."""

import argparse
import time
from pathlib import Path

import torch

from data.element_inputs import HEAD, RECIPE, training_features
from evaluation.element_controls import (
    evaluate_controls,
    evaluate_features,
    evaluate_seeds,
)
from execution import start_run
from grounding.element_scoring import (
    ElementScorer,
    LinearScorer,
    SubsetReranker,
    load_model,
)
from grounding.element_training import train_head
from manifest import CODE_ROOT, load


def arguments(name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start_run(args.output, CODE_ROOT / "configs/element.json", name, vars(args))
    return args.output


def controls_main():
    out = arguments("element-controls")
    started = time.monotonic()
    linear, training = train_head(
        out / "linear", training_features(), LinearScorer, RECIPE["seed"]
    )
    evaluate_controls(
        out,
        load(),
        dict(original=load_model(HEAD), linear=linear),
        dict(linear=training),
        started,
    )


def features_main():
    out = arguments("element-features")
    started = time.monotonic()
    dataset = training_features()
    models, training = {}, {}
    for name, keep in {**RECIPE["feature_subsets"], "full": None}.items():
        models[name], training[name] = {}, {}
        for seed in RECIPE["seeds"]:
            factory = (
                ElementScorer
                if keep is None
                else lambda keep=keep: SubsetReranker(keep)
            )
            models[name][seed], training[name][seed] = train_head(
                out / name / str(seed), dataset, factory, seed
            )
    evaluate_features(out, models, training, started)


def seeds_main():
    out = arguments("element-seeds")
    started = time.monotonic()
    dataset = training_features()
    models, training = {}, {}
    for seed in RECIPE["seeds"]:
        models[seed], training[seed] = train_head(
            out / str(seed), dataset, ElementScorer, seed
        )
    evaluate_seeds(out, models, training, started)
