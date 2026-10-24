"""Coordinate inference helpers."""
from types import SimpleNamespace
import pytest
import torch
import grounding.coordinates as coordinates


def test_coordinate_prefix_preserves_the_exact_full_token_sequence():
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))
    cache = tokenizer.encode(coordinates.COORDINATE_CACHE_PREFIX)
    suffix = coordinates.coordinate_suffix_tokens(tokenizer)
    assert cache + suffix == tokenizer.encode(coordinates.encoding.A_PREFIX)
    broken = SimpleNamespace(encode=lambda text, **kw: [len(text)])
    with pytest.raises(ValueError, match='not a token prefix'):
        coordinates.coordinate_suffix_tokens(broken)




def test_coordinate_state_uses_requested_checkpoint_without_loading_a_real_model(monkeypatch, tmp_path):
    import peft
    import fara as package
    import grounding.checkpoints as checkpoints
    processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text, **kw: list(text.encode())))
    model = torch.nn.Linear(1, 1)
    seen = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(package, 'load_model', lambda: (processor, model))
    monkeypatch.setattr(checkpoints, 'load_scoring_state', lambda path: seen.append(path) or {'adapter': {}})
    monkeypatch.setattr(peft, 'set_peft_model_state_dict', lambda model, state: None)
    state = coordinates.load_state(checkpoint=tmp_path)
    assert seen == [tmp_path] and state['model'] is model and not model.training
