"""Prepare element features, train a head, then evaluate it."""

import argparse
from pathlib import Path

import torch

from data import element_inputs as data
from evaluation.element_results import evaluate_tests
from execution import gpu_lock, start_run
from grounding.element_training import train
from manifest import CODE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data.TRAIN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    start_run(
        args.output, CODE_ROOT / "configs/element.json", "element-training", vars(args)
    )
    dataset = data.prepare(args.output, args.source)
    with gpu_lock(args.device):
        training = train(args.output, dataset, args.source, device=args.device)
    evaluate_tests(args.output, training)
