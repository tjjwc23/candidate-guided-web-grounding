"""Orchestrate evidence diagnostics and matched candidate-policy training."""

import argparse
import time
from pathlib import Path

import torch

from data.element_inputs import (
    HEAD,
    SEEDS,
    evidence_rows,
    functional_training_data,
    inputs,
)
from evaluation.element_mechanisms import (
    evaluate_functional,
    evaluate_mechanisms,
)
from execution import start_run
from grounding.element_scoring import load_model
from grounding.element_training import fit_native_head, tune
from manifest import CODE_ROOT, load


def arguments(name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start_run(args.output, CODE_ROOT / "configs/element.json", name, vars(args))
    return args.output


def mechanisms_main():
    out = arguments("element-mechanisms")
    evaluate_mechanisms(out, load(), time.monotonic())


def functional_main():
    out = arguments("element-functional")
    started = time.monotonic()
    reference = load()
    rows, baseline, data, _, points, semantics = inputs("valid", reference)
    ev, _, _ = evidence_rows(rows, points, semantics, data)
    coeff = tune(rows, baseline, ev, out)
    data, restricted, labels = functional_training_data()
    models, training = {"main": load_model(HEAD)}, {}
    for seed in SEEDS:
        for policy, dataset in (("all", data), ("min", restricted)):
            name = f"{policy}_{seed}"
            models[name], training[name] = fit_native_head(
                dataset, labels, baseline, seed, out / name
            )
    evaluate_functional(out, reference, coeff, models, training, started)
