"""One inference export manifest; training resume state is never an inference input."""
from pathlib import Path

from artifacts import file_digest, read_json, write_json

FORMAT = "stepengine-model-v1"


def seal_export(directory, *, kind, contract, files, metadata):
    """Write the manifest last, after the owner has saved all model files."""
    directory = Path(directory)
    value = dict(
        format=FORMAT, kind=kind, contract=contract, metadata=metadata,
        files={name: file_digest(directory / name) for name in sorted(files)},
    )
    write_json(directory / "model.json", value, atomic=True)
    return value


def read_export(directory, *, kind, contract):
    """Require this format and verify every exported file; never inspect old runs."""
    directory = Path(directory)
    value = read_json(directory / "model.json")
    if (value.get("format"), value.get("kind"), value.get("contract")) != (FORMAT, kind, contract):
        raise ValueError(f"Expected {FORMAT} {kind}/{contract}: {directory}")
    if not value["files"]:
        raise ValueError(f"Empty model export: {directory}")
    for name, digest in value["files"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Model file changed: {directory / name}")
    return value
