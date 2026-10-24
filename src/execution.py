"""execution."""

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from artifacts import content_manifest, write_json
from artifacts import file_digest as digest
from artifacts import read_json as read


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)








def experiment_status(output, stage, **fields):
    """Persist a stage transition and emit it to the process log."""
    value = dict(
        stage=stage,
        pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat(),
        **fields,
    )
    write_json(Path(output) / "status.json", value, atomic=True, mkdir=True)
    print(json.dumps(value), flush=True)


def start_run(output, recipe_path, name, arguments):
    """Record the static recipe and invocation once; never synthesize trainer config."""
    from manifest import CODE_ROOT

    output = Path(output)
    value = dict(
        experiment=name,
        recipe=read(recipe_path),
        recipe_sha256=digest(recipe_path),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()
        },
        source=content_manifest(sorted((CODE_ROOT / "src").rglob("*.py"))),
    )
    path = output / "run.json"
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Run inputs changed: {path}; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    return value


@contextmanager
def gpu_lock(device):
    """Small-head CPU jobs bypass the shared GPU queue."""
    import fcntl

    from manifest import ROOT

    if device == "cpu":
        yield
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this stage")
    lock_path = ROOT / "outputs/.gpu0.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield




