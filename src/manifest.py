"""manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WEBAGENT_REPO", CODE_ROOT))


CONFIG_PATH = CODE_ROOT / "configs/paper.json"


def load(path: Path | None = None) -> dict:
    return json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def weight(name: str, config: dict | None = None) -> dict:
    return (config or load())["frozen_weights"][name]


def checkpoint_dir(name: str, config: dict | None = None) -> Path:
    return resolve(weight(name, config)["checkpoint_dir"])


def frozen_input(
    name: str, split: str | None = None, config: dict | None = None
) -> Path:
    inputs = (config or load())["frozen_inputs"]
    node = inputs[name]
    if isinstance(node, dict):
        assert split is not None, f"{name} needs a split"
        node = node[split]
    return resolve(node)


def predictions(split: str, config: dict | None = None) -> Path:
    return resolve((config or load())["frozen_predictions"][split]["path"])


def reported(split: str, config: dict | None = None) -> dict:
    return (config or load())["reported"][split]


def _digest(path: Path) -> str:
    """Stream files, retaining the existing directory digest convention."""
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else [f for f in sorted(path.rglob("*")) if f.is_file()]
    )
    for f in files:
        if path.is_dir():
            digest.update(f.name.encode())
        with f.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_inputs(config: dict | None = None) -> list[dict]:
    """Check every configured input/prediction path exists and matches its recorded hash."""
    config = config or load()
    out = []
    for name, entry in config["frozen_inputs"].items():
        items = entry if isinstance(entry, dict) else {"_": entry}
        for split, path in items.items():
            out.append((f"{name}/{split}", path))
    for name, node in config["frozen_predictions"].items():
        out.append((f"predictions/{name}", node["path"]))
    rows = []
    for key, path in out:
        p = resolve(path)
        recorded = config.get("input_sha256", {}).get(key)
        if not p.exists():
            rows.append({"name": key, "path": str(p), "status": "missing"})
        elif not recorded:
            rows.append({"name": key, "path": str(p), "status": "unrecorded"})
        elif _digest(p) != recorded:
            rows.append({"name": key, "path": str(p), "status": "hash-mismatch"})
        else:
            rows.append({"name": key, "path": str(p), "status": "ok"})
    return rows


def verify(config: dict | None = None) -> list[dict]:
    """Return one record per frozen weight and frozen input: present/absent and hash match."""
    config = config or load()
    out = []
    for name, entry in config["frozen_weights"].items():
        path = resolve(entry["path"])
        if not path.exists():
            out.append({"name": name, "path": str(path), "status": "missing"})
            continue
        digest = _digest(path)
        out.append(
            {
                "name": name,
                "path": str(path),
                "status": "ok" if digest == entry["sha256"] else "hash-mismatch",
                "sha256": digest[:16],
            }
        )
    out.extend(verify_inputs(config))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="verify cached reproduction inputs and predictions, without requiring model weights",
    )
    parser.add_argument("--print", dest="dump", action="store_true")
    args = parser.parse_args()
    config = load()
    if args.verify or args.verify_inputs:
        rows = verify_inputs(config) if args.verify_inputs else verify(config)
        for row in rows:
            print("%-10s %-24s %s" % (row["status"], row["name"], row["path"]))
        return 0 if all(r["status"] == "ok" for r in rows) else 1
    if args.dump:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    print(
        json.dumps(
            {"recipe": config["recipe"]["name"], "reported": config["reported"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_recipe(name):
    """Read a checked-in recipe; run directories are never configuration sources."""
    return load(CODE_ROOT / "configs" / (name + ".json"))


def element_head():
    """The paper's 18-feature element head."""
    return resolve(load_recipe("element")["inference"]["head"])


def grounding_inputs(config=None):
    config = load() if config is None else config
    return {
        split: frozen_input("anchored_rows", split, config).parent
        for split in config["splits"]
    }


def active_model(name):
    """Configured model export for coordinate inference."""
    return resolve(load_recipe("models")[name])
