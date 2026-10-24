"""artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, indent=2, atomic=True, mkdir=True, newline=False):
    """Atomically publish JSON, creating parent directories by default."""
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_suffix(path.suffix + ".tmp") if atomic else path
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent)
        + ("\n" if newline else ""),
        encoding="utf-8",
    )
    if atomic:
        destination.replace(path)




def content_manifest(paths):
    """Hash required files/directories with full identities and bounded memory."""
    result = {}
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        files = (
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
        result[str(path)] = {
            str(p.relative_to(path)) if path.is_dir() else ".": file_digest(p)
            for p in files
        }
    return result


