"""Relocated evidence must retain the identity and scores of measured inputs."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_submission_latency",
    Path(__file__).resolve().parents[1] / "papers/verify_submission_latency.py",
)
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


@pytest.mark.parametrize("saved_key", ["logits", "semantic_logits"])
def test_relocated_semantic_evidence_matches_by_content(tmp_path, saved_key):
    saved, configured = tmp_path / "old.json", tmp_path / "new.json"
    saved.write_text(json.dumps({"action_key": "task:1", saved_key: [0.2, -0.3]}))
    configured.write_text(json.dumps({"action_key": "task:1", "semantic_logits": [0.2, -0.3]}))
    latency.verify_semantic_evidence(saved, configured, "task:1")


@pytest.mark.parametrize("action_key,scores", [("other:1", [0.2, -0.3]), ("task:1", [0.2, -0.4])])
def test_relocated_semantic_evidence_rejects_changed_content(tmp_path, action_key, scores):
    saved, configured = tmp_path / "old.json", tmp_path / "new.json"
    saved.write_text(json.dumps({"action_key": "task:1", "logits": [0.2, -0.3]}))
    configured.write_text(json.dumps({"action_key": action_key, "semantic_logits": scores}))
    with pytest.raises(AssertionError):
        latency.verify_semantic_evidence(saved, configured, "task:1")
