import torch
import pytest
from grounding.element_scoring import ElementScorer, LinearScorer, SubsetReranker, save_model, load_model
from grounding.element_features import FEATURE_NAMES

def test_subset_starts_from_native_full_head_without_constant_folding():
    keep = [
        FEATURE_NAMES.index("semantic_logit"),
        FEATURE_NAMES.index("window_document_top"),
    ]
    torch.manual_seed(31)
    full = ElementScorer()
    torch.manual_seed(31)
    subset = SubsetReranker(keep)
    assert torch.equal(subset.correction[0].weight, full.correction[0].weight[:, keep])
    assert torch.equal(subset.correction[0].bias, full.correction[0].bias)
    assert sum(p.numel() for p in full.parameters()) == 2440
    assert sum(p.numel() for p in LinearScorer().parameters()) == 18

@pytest.mark.parametrize('factory', [ElementScorer, LinearScorer, lambda: SubsetReranker([0, 11])])
def test_native_export_roundtrip_keeps_architecture_and_tensors(tmp_path, factory):
    model = factory()
    path = tmp_path / 'head.pt'
    save_model(model, path)
    loaded = load_model(path)
    assert type(loaded) is type(model)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())


def test_raw_and_19d_weights_are_not_implicitly_converted(tmp_path):
    path = tmp_path / 'head.pt'
    torch.save(ElementScorer().state_dict(), path)
    with pytest.raises(FileNotFoundError):
        load_model(path)
    save_model(ElementScorer(), path)
    state = torch.load(path, weights_only=True)
    state['correction.0.weight'] = torch.zeros(40, 19)
    torch.save(state, path)
    from artifacts import read_json, write_json, file_digest
    meta = read_json(tmp_path / 'model.json');meta['files']['head.pt'] = file_digest(path)
    write_json(tmp_path / 'model.json', meta)
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_model(path)
