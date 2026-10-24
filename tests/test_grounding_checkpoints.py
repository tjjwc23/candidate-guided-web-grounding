"""Coordinate inference reads sealed exports, never training state or legacy formats."""
import json
import torch
import pytest
from safetensors.torch import save_file
from grounding.checkpoints import export_model, load_scoring_state


def export(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    save_file({'weight': torch.tensor([1., 2.])}, adapter / 'adapter_model.safetensors')
    (adapter / 'adapter_config.json').write_text('{}')
    export_model(tmp_path, epoch=2)


def test_inference_ignores_training_state_and_does_not_deserialize_optimizer(tmp_path, monkeypatch):
    export(tmp_path)
    (tmp_path / 'checkpoint.pt').write_bytes(b'unreadable training state')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Must not load a training checkpoint'))
    state = load_scoring_state(tmp_path)
    assert state['epoch'] == 2
    assert torch.equal(state['adapter']['weight'], torch.tensor([1., 2.]))


@pytest.mark.parametrize('old_file', ['summary.json', 'run_protocol.json', 'model.json'])
def test_old_layouts_do_not_trigger_compatibility(tmp_path, old_file):
    (tmp_path / old_file).write_text(json.dumps({'epochs': 2, 'adapter_sha256': 'old'}))
    with pytest.raises((FileNotFoundError, ValueError)):
        load_scoring_state(tmp_path)


def test_export_rejects_modified_weights_or_contract(tmp_path):
    export(tmp_path)
    p = tmp_path / 'model.json'
    meta = json.loads(p.read_text());meta['contract'] = 'unknown';p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Expected'):
        load_scoring_state(tmp_path)
    export_model(tmp_path, epoch=2)
    (tmp_path / 'adapter/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_scoring_state(tmp_path)


@pytest.mark.integration
def test_published_coordinate_export_preserves_tensor_values():
    from manifest import active_model, checkpoint_dir
    from safetensors.torch import load_file
    current = load_scoring_state(active_model('coordinate'))['adapter']
    original = load_file(checkpoint_dir('grounding_c50_adapter') / 'adapter/adapter_model.safetensors')
    assert current.keys() == original.keys()
    assert all(torch.equal(current[k], original[k]) for k in current)
