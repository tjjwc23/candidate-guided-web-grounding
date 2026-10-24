import json
from types import SimpleNamespace
import pytest
import torch
from data.dom import canonical_candidates; from data.dom import candidate_order_identity
from retrieval.inputs import proposal_indices; from retrieval.inputs import load_proposal_cache; from retrieval.inputs import PROPOSAL_CACHE_KIND


def test_order_does_not_depend_on_labels_or_input_permutation():
    rows = [
        dict(backend_node_id=str(i), text=str(i), label=int(i == 2)) for i in range(10)
    ]
    first = canonical_candidates(rows)
    changed = canonical_candidates(
        [dict(r, label=1 - r["label"]) for r in reversed(rows)]
    )
    assert [r["backend_node_id"] for r in first] == [
        r["backend_node_id"] for r in changed
    ]
    assert [r["text"] for r in first] == [r["text"] for r in changed]


def test_stale_proposal_indices_are_rejected(tmp_path):
    sample = dict(backend_node_ids=["10", "20"], dom_texts=["a", "b"])
    cache = dict(
        kind=PROPOSAL_CACHE_KIND,
        actions={
            "a": dict(
                indices=[1, 0],
                candidate_order_identity=candidate_order_identity(sample),
            )
        },
    )
    assert proposal_indices(cache, "a", sample) == [1, 0]
    with pytest.raises(ValueError, match="identity/order"):
        proposal_indices(
            cache, "a", dict(backend_node_ids=["20", "10"], dom_texts=["b", "a"])
        )
    file = tmp_path / "old.json"
    file.write_text(json.dumps(dict(kind="reranker.deberta_proposals.v2")))
    with pytest.raises(ValueError, match="Unsupported"):
        load_proposal_cache(file)


def test_proposal_generation_keeps_all_negative_actions_and_ignores_relabeling(tmp_path, monkeypatch):
    import retrieval.deberta as deberta

    sample = dict(backend_node_ids=['10', '20', '30'], labels=[0, 0, 0], query='task')
    records = [dict(annotation_id='task', action_uid='action', sample=sample),
               dict(annotation_id='task', action_uid='no_candidates', sample=None)]
    args = SimpleNamespace(top_k=2, max_length=512, encoder_batch_size=4,
        preprocess_workers=1, history_steps=4, seed=42, data_path='local', dataset_split='test_task',
        deberta_model='local_model', output=str(tmp_path/'proposals.json'))

    class FakeModel:
        def to(self, device): return self
        def eval(self): return self

    monkeypatch.setattr(deberta, 'parse_args', lambda: args)
    monkeypatch.setattr(deberta.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(deberta, 'set_seed', lambda seed: None)
    monkeypatch.setattr(deberta, 'load_action_records', lambda *a: records)
    monkeypatch.setattr(deberta, 'build_dom_candidate_sample', lambda record, **kw: record['sample'])
    monkeypatch.setattr(deberta, '_iter_proposal_candidate_texts',
        lambda records, prepared, workers: iter((row, ['a', 'b', 'c']) for row in prepared))
    monkeypatch.setattr(deberta, 'resolve_local_model', lambda path: path)
    monkeypatch.setattr(deberta.AutoTokenizer, 'from_pretrained', lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(deberta.AutoModelForSequenceClassification, 'from_pretrained',
        lambda *a, **kw: FakeModel())
    monkeypatch.setattr(deberta, 'tokenize_candidates', lambda tokenizer, batch, length: batch)
    monkeypatch.setattr(deberta, 'score_candidates',
        lambda model, batch, device, size: torch.tensor([.2, .9, .9]))

    deberta.main()
    negative = load_proposal_cache(args.output)
    assert negative['actions']['task_action']['indices'] == [1, 2]
    assert 'task_no_candidates' not in negative['actions']
    assert negative['metrics']['actions'] == 1
    assert negative['metrics']['gold_recalled'] == 0

    sample['labels'] = [1, 1, 1]
    deberta.main()
    positive = load_proposal_cache(args.output)
    assert negative['actions'] == positive['actions']
    assert positive['metrics']['gold_recalled'] == 1


def test_full_proposal_rebuild_streams_without_images_and_resumes(
    tmp_path, monkeypatch
):
    import torch
    from types import SimpleNamespace
    import manifest
    from artifacts import file_digest, read_json
    from data import grounding_inputs
    from retrieval import deberta

    weight = tmp_path / "pytorch_model.bin"
    weight.write_bytes(b"test-weight")
    sample = dict(
        backend_node_ids=["1", "2", "3"],
        dom_texts=["a", "b", "c"],
        labels=[0, 0, 0],
        query="task",
    )
    raw = [
        dict(
            annotation_id="t",
            action_uid=str(i),
            cleaned_html="html",
            sample=sample if i else None,
        )
        for i in range(3)
    ]

    def records(split, columns):
        assert "screenshot" not in columns
        yield from raw

    class FakeModel:
        def cuda(self):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(grounding_inputs, "raw_records", records)
    monkeypatch.setattr(
        manifest, "weight", lambda name: dict(sha256=file_digest(weight))
    )
    monkeypatch.setattr(deberta, "resolve_local_model", lambda name: str(tmp_path))
    monkeypatch.setattr(
        deberta, "build_dom_candidate_sample", lambda r, **kw: r["sample"]
    )
    monkeypatch.setattr(
        deberta, "format_proposal_candidate_texts", lambda *a: ["a", "b", "c"]
    )
    monkeypatch.setattr(
        deberta.AutoTokenizer, "from_pretrained", lambda *a, **kw: SimpleNamespace()
    )
    monkeypatch.setattr(
        deberta.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *a, **kw: FakeModel(),
    )
    monkeypatch.setattr(deberta, "tokenize_candidates", lambda *a: None)
    monkeypatch.setattr(
        deberta, "score_candidates", lambda *a: torch.tensor([0.2, 0.9, 0.9])
    )
    output = tmp_path / "out"
    deberta.build_streaming_cache(output, "train")
    cache = read_json(output / "proposals.json")
    assert cache["metrics"] == dict(raw_actions=3, actions=2)
    assert set(cache["actions"]) == {"t_1", "t_2"}
    assert cache["actions"]["t_1"]["indices"] == [1, 2, 0]

    def unexpected(*args):
        raise AssertionError("Completed Proposal actions must not score again")

    monkeypatch.setattr(deberta, "score_candidates", unexpected)
    deberta.build_streaming_cache(output, "train")
    assert read_json(output / "proposals.json") == cache
