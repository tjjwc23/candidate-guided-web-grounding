"""Write resumable frozen-coordinate evidence; model execution owns decoding."""

import argparse
import gc
import time
from pathlib import Path

import torch
from PIL import Image

import artifacts, manifest as mainline
from artifacts import write_json
from execution import experiment_status as status, gpu_lock
from grounding.coordinates import greedy_coordinate
from manifest import ROOT


def coordinates(output, rows, *, checkpoint=None):
    from grounding.coordinates import load_state

    pending = [
        r for r in rows if not (output / "points" / f"{r['index']}.json").exists()
    ]
    if not pending:
        return
    checkpoint = checkpoint or mainline.active_model("coordinate")
    state = load_state(checkpoint=checkpoint)
    state["model"].requires_grad_(False)
    started = time.monotonic()
    prior = len(rows) - len(pending)
    for number, row in enumerate(pending, 1):
        torch.cuda.reset_peak_memory_stats()
        action_started = time.monotonic()
        windows = []
        for window in row["windows"]:
            value = greedy_coordinate(
                state["processor"],
                state["model"],
                dict(task=row["task"], history=row["history"], crop=window["crop"]),
                state["suffix"],
            )
            with Image.open(ROOT / window["crop"]) as image:
                size = list(image.size)
            windows.append(dict(value, top=window["top"], crop_size=size))
        torch.cuda.synchronize()
        write_json(
            output / "points" / f"{row['index']}.json",
            dict(
                index=row["index"],
                action_key=row["action_key"],
                windows=windows,
                seconds=time.monotonic() - action_started,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            ),
        )
        if len(rows) <= 3 or number == 1 or number % 50 == 0 or number == len(pending):
            elapsed = time.monotonic() - started
            status(
                output,
                "coordinate_inference",
                completed=prior + number,
                total=len(rows),
                new_completed=number,
                new_total=len(pending),
                seconds_per_action=elapsed / number,
                eta_minutes=(len(pending) - number) * elapsed / number / 60,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
    del state
    gc.collect()
    torch.cuda.empty_cache()


def prepare_coordinates(args, rows):
    checkpoint_hash = artifacts.file_digest(
        mainline.active_model("coordinate")
        / "adapter/adapter_model.safetensors"
    )
    input_hash = artifacts.file_digest(
        mainline.grounding_inputs()[args.split] / "rows.json"
    )
    cached = 0
    if args.coordinate_cache is not None:
        cache = args.coordinate_cache
        previous = artifacts.read_json(cache / "protocol.json")
        assert previous["checkpoint_sha256"] == checkpoint_hash
        assert previous["input_sha256"] == input_hash
        for row in rows:
            source = cache / "points" / f"{row['index']}.json"
            destination = args.output / "points" / f"{row['index']}.json"
            if not source.exists():
                continue
            record = artifacts.read_json(source)
            assert record["action_key"] == row["action_key"]
            assert len(record["windows"]) == len(row["windows"])
            for saved, window in zip(record["windows"], row["windows"]):
                with Image.open(ROOT / window["crop"]) as image:
                    assert saved["crop_size"] == list(image.size)
                assert saved["top"] == window["top"]
            cached += 1
            if not destination.exists():
                write_json(
                    destination,
                    dict(record, coordinate_source=str(source), reused=True),
                )
    protocol = dict(
        split=args.split,
        checkpoint=str(mainline.active_model("coordinate")),
        checkpoint_sha256=checkpoint_hash,
        inputs=str(mainline.grounding_inputs()[args.split]),
        input_sha256=input_hash,
        actions=[r["action_key"] for r in rows],
        cached_actions=cached,
        method="Greedy native coordinate evidence on frozen Top20-anchored C50 inputs",
        model_inference=True,
        training=False,
        browser_used=False,
        raw_dump_read=False,
        max_coordinate_tokens=24,
        precision="BF16",
        cache="Reuse only coordinates with matching checkpoint, input hash, action identity, window top and crop dimensions; no replay or hit caches used for inference",
    )
    path = args.output / "protocol.json"
    if path.exists():
        assert (
            artifacts.read_json(path) == protocol
        ), "Changed protocol requires a new output directory"
    else:
        write_json(path, protocol)
    manifest = [
        dict(
            action_key=r["action_key"],
            index=r["index"],
            task=r["task"],
            history=r["history"],
            windows=[dict(crop=w["crop"], top=w["top"]) for w in r["windows"]],
        )
        for r in rows
    ]
    write_json(args.output / "inference_inputs.json", manifest)
    return manifest


def generate(args, manifest):
    pending = [
        r
        for r in manifest
        if not (args.output / "points" / f"{r['index']}.json").exists()
    ]
    if args.smoke and pending:

        def maximum_pixels(row):
            sizes = []
            for window in row["windows"]:
                with Image.open(ROOT / window["crop"]) as image:
                    sizes.append(image.width * image.height)
            return max(sizes)

        widest = max(pending, key=maximum_pixels)
        pending = list({r["action_key"]: r for r in (pending[0], widest)}.values())
    if not pending:
        return
    status(args.output, "waiting_for_gpu", pending=len(pending), total=len(manifest))
    with gpu_lock("cuda"):
        coordinates(args.output, pending if args.smoke else manifest)


def coordinate_evidence_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=tuple(mainline.grounding_inputs()), default="test_task"
    )
    parser.add_argument(
        "--coordinate-cache",
        type=Path,
        help="Optional native point cache with matching checkpoint and geometry",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        artifacts.read_json(mainline.grounding_inputs()[args.split] / "rows.json"),
        key=lambda r: r["index"],
    )
    manifest = prepare_coordinates(args, rows)
    generate(args, manifest)
    if args.smoke:
        status(
            args.output,
            "smoke_complete",
            completed=sum(
                (args.output / "points" / f"{r['index']}.json").exists() for r in rows
            ),
            total=len(rows),
        )
    else:
        completed = sum(
            (args.output / "points" / f"{r['index']}.json").exists() for r in rows
        )
        assert completed == len(rows), "Coordinate evidence is incomplete"
        summary = dict(
            status="complete",
            split=args.split,
            actions=completed,
            training=False,
            scope="Coordinate cache only; use infer grounding for unified element decisions",
        )
        write_json(args.output / "summary.json", summary)
        status(
            args.output,
            "complete",
            **{k: v for k, v in summary.items() if k != "status"},
        )
