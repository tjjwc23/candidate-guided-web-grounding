"""data / screenshots."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

from PIL import Image

from data.geometry import display_point_coordinate
from manifest import ROOT as REPOSITORY_ROOT

# Offline dataset/model mode must be set before datasets/transformers import.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


# Mind2Web screenshots can be very tall full-page captures; Pillow's default
# decompression-bomb limit (178M pixels) can trip on them.  This is a local,
# trusted dataset, so lift the limit like the Reranker probes effectively do.
Image.MAX_IMAGE_PIXELS = None


def decode_screenshot(value) -> Image.Image:
    """Decode a dataset screenshot cell into an RGB PIL image.

    The cell may already be a PIL image, a dict with bytes/path, or a path.
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert("RGB")
    raise ValueError(f"Unsupported screenshot value: {type(value)!r}")


"""Label-independent multi-window selection for Coordinate Grounding."""

WINDOW_HEIGHT = 720
MAX_WINDOWS = 3
PROJECT_ROOT = REPOSITORY_ROOT


def greedy_window_tops(boxes, page_height):
    """Select up to three 720px windows that cover candidate centres."""
    valid = []
    for candidate, box in boxes.items():
        if box and float(box[2]) > 0 and float(box[3]) > 0:
            valid.append(
                (
                    candidate,
                    float(box[0]) + float(box[2]) / 2,
                    float(box[1]) + float(box[3]) / 2,
                )
            )
    maximum_top = max(0, page_height - WINDOW_HEIGHT)
    possible = sorted(
        {0}
        | {max(0, min(int(y - WINDOW_HEIGHT / 2), maximum_top)) for _, _, y in valid}
    )
    remaining = list(valid)
    selected = []
    for _ in range(MAX_WINDOWS):
        if not remaining:
            break
        top = max(
            possible,
            key=lambda value: sum(
                value <= y < value + WINDOW_HEIGHT for _, _, y in remaining
            ),
        )
        covered = [item for item in remaining if top <= item[2] < top + WINDOW_HEIGHT]
        if not covered:
            break
        selected.append((top, covered))
        remaining = [item for item in remaining if item not in covered]
    return selected


def prepare_candidate_windows(
    page,
    boxes,
    output: Path,
    index: int,
    *,
    path_root: Path | None = PROJECT_ROOT,
    layout_boxes=None,
):
    """Crop the label-neutral windows consumed by Coordinate Grounding.

    Dataset preparation stores paths relative to the artifact root. Live
    callers can pass ``path_root=None`` to keep absolute temporary paths.
    ``layout_boxes`` selects the layout; all ``boxes`` can join its windows.
    """
    page_width, page_height = page.size
    selected = greedy_window_tops(boxes if layout_boxes is None else layout_boxes, page_height)
    (output / "crops").mkdir(parents=True, exist_ok=True)
    windows = []
    assigned = set()
    for window_index, (top, _) in enumerate(selected):
        crop = page.crop((0, top, page_width, min(page_height, top + WINDOW_HEIGHT)))
        crop_width, crop_height = crop.size
        centres = {}
        for candidate, box in boxes.items():
            if not box or candidate in assigned:
                continue
            x, y, width, height = map(float, box[:4])
            if width <= 0 or height <= 0:
                continue
            centre_x = x + width / 2
            centre_y = y + height / 2 - top
            if not (0 <= centre_x < crop_width and 0 <= centre_y < crop_height):
                continue
            centres[str(candidate)] = [
                display_point_coordinate(centre_x, crop_width),
                display_point_coordinate(centre_y, crop_height),
            ]
        if not centres:
            crop.close()
            continue
        crop_path = output / "crops" / f"{index}_{window_index}.png"
        crop.save(crop_path)
        crop.close()
        assigned.update(map(int, centres))
        stored_crop = (
            crop_path if path_root is None else crop_path.relative_to(path_root)
        )
        windows.append(
            {
                "top": int(top),
                "crop": str(stored_crop),
                "centres": centres,
            }
        )
    return windows


ROOT = REPOSITORY_ROOT




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_website")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/pages")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from data.mind2web import action_key, load_action_records

    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_action_records("osunlp/Multimodal-Mind2Web", args.split)
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    written = skipped = 0
    for index in range(total):
        row = dataset[index]
        key = action_key(row)
        target = args.output / f"{key}.png"
        if target.exists():
            skipped += 1
        else:
            shot = row.get("screenshot")
            if shot is None:
                skipped += 1
                continue
            image = shot if isinstance(shot, Image.Image) else Image.open(io.BytesIO(shot))
            image.convert("RGB").save(target)
            image.close()
            written += 1
        if (index + 1) % 100 == 0:
            print(json.dumps({"stage": "pages", "completed": index + 1, "total": total,
                              "written": written, "skipped": skipped}), flush=True)
    print(json.dumps({"stage": "pages_complete", "total": total, "written": written,
                      "skipped": skipped}), flush=True)
    return 0
