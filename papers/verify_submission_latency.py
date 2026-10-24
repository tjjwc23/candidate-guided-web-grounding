"""Independently audit saved latency measurements; never rerun GPU inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter
from zipfile import ZipFile

from manifest import ROOT, CODE_ROOT
SOURCE_ARCHIVE = CODE_ROOT / 'papers/measurement-sources-20261022.zip'
RUN = ROOT / 'outputs/ch04_submission_controls_20261022/latency_v2'
read = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def semantic_scores(record):
    """Read recorded evidence from either historical schema during this audit."""
    return record['semantic_logits'] if 'semantic_logits' in record else record['logits']


def verify_semantic_evidence(saved_path, configured_path, action_key):
    assert saved_path.is_file() and configured_path.is_file()
    saved, configured = read(saved_path), read(configured_path)
    assert saved['action_key'] == configured['action_key'] == action_key
    assert semantic_scores(saved) == semantic_scores(configured), (saved_path, configured_path)


def audit():
    import torch
    from PIL import Image
    from grounding.element_scoring import ElementScorer as JointReranker; from grounding.element_features import features as features; from grounding.element_scoring import predict as predict

    protocol, samples, summary = [read(RUN / f'{n}.json') for n in ('protocol', 'samples', 'summary')]
    config = read(CODE_ROOT / 'configs/paper.json')['frozen_inputs']
    assert summary['status'] == read(RUN / 'status.json')['status'] == 'complete'
    assert len(samples) == protocol['samples'] == 120 and protocol['repeats'] == 2
    with ZipFile(SOURCE_ARCHIVE) as archive:
        prefix = 'latency-source-20261022/'
        provenance = json.loads(archive.read(prefix + 'provenance.json'))
        for name, expected in protocol['source_sha256'].items():
            source = provenance[name]
            actual = hashlib.sha256(archive.read(prefix + source['file'])).hexdigest()
            assert actual == source['sha256'] == expected, name
    for name, key in [('outputs/mainline/grounding/adapter/adapter_model.safetensors', 'adapter_sha256'),
                      ('outputs/ch04_element18_20261021/main/head.pt', 'head_sha256')]:
        assert digest(ROOT / name) == protocol[key], name
    assert summary['environment'] == read(RUN / 'environment.json')['stable']
    for split in ('test_task', 'test_website', 'test_domain'):
        rows_path = ROOT / config['anchored_rows'][split]
        pages_path = ROOT / config['grounding_page_index'][split]
        assert digest(rows_path) == protocol['sources'][split]['rows']
        assert digest(pages_path) == protocol['sources'][split]['pages']
        rows = sorted(read(rows_path), key=lambda r: hashlib.sha256(
            f'{protocol["seed"]}:{split}:{r["action_key"]}'.encode()).hexdigest())[:40]
        selected = [s for s in samples if s['split'] == split]
        assert len(selected) == 40
        pages = {p['action_key']: p['page_size'] for p in map(json.loads, pages_path.read_text().splitlines())}
        for saved, original in zip(selected, rows):
            for key in ('index', 'action_key', 'task_id', 'task', 'history', 'candidate_bboxes', 'windows'):
                assert saved[key] == original[key], (split, key)
            assert saved['page_size'] == pages[saved['action_key']]
            verify_semantic_evidence(
                ROOT / saved["semantic_path"],
                ROOT / config["grounding_semantics"][split] / f'{saved["index"]}.json',
                saved["action_key"],
            )

    torch.set_num_threads(4)
    head = JointReranker().eval()
    head.load_state_dict(torch.load(ROOT / 'outputs/ch04_element18_20261021/main/head.pt',
                                   map_location='cpu', weights_only=True))
    records, expected_names = {}, set()
    cached_coordinate_matches = 0
    input_hashes = {}
    for j, sample in enumerate(samples):
        sem_path = ROOT / sample['semantic_path']
        sem = read(sem_path)
        assert sem['action_key'] == sample['action_key']
        input_hashes[str(sem_path.relative_to(ROOT))] = digest(sem_path)
        for window in sample['windows']:
            path = ROOT / window['crop']
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
            with Image.open(path) as im:
                assert list(im.size) == [sample['page_size'][0], min(720, sample['page_size'][1] - window['top'])]
        for repeat in range(2):
            for mode in ('single', 'multi'):
                name = f'{j:04d}_{repeat}_{mode}.json'
                expected_names.add(name)
                record = read(RUN / 'records' / name)
                for key, value in dict(sample_index=j, repeat=repeat, mode=mode,
                                       action_key=sample['action_key'], split=sample['split']).items():
                    assert record[key] == value, (name, key)
                windows = sample['windows'][:1] if mode == 'single' else sample['windows']
                assert record['windows'] == len(record['predictions']) == len(windows)
                for key in ('coordinate_seconds', 'feature_head_seconds', 'total_seconds',
                            'peak_allocated_gib', 'peak_reserved_gib'):
                    assert math.isfinite(record[key]) and record[key] > 0
                assert math.isclose(record['total_seconds'], record['coordinate_seconds'] + record['feature_head_seconds'], abs_tol=1e-12)
                assert record['peak_allocated_gib'] <= record['peak_reserved_gib'] <= summary['environment']['total_memory_gib']
                feat = features(dict(sample, windows=windows),
                                dict(action_key=sample['action_key'], windows=record['predictions']),
                                semantic_scores(sem))
                decisions, _ = predict(head, [feat], 'cpu')
                assert decisions[sample['action_key']] == record['winner'], name
                records[j, repeat, mode] = record
            assert records[j, repeat, 'single']['predictions'][0] == records[j, repeat, 'multi']['predictions'][0]
        for mode in ('single', 'multi'):
            assert records[j, 0, mode]['predictions'] == records[j, 1, mode]['predictions']
        cached = read(ROOT / config['grounding_points'][sample['split']] / f'{sample["index"]}.json')
        assert [p['coordinate'] for p in cached['windows']] == [p['coordinate'] for p in records[j, 0, 'multi']['predictions']]
        cached_coordinate_matches += 1
    assert {p.name for p in (RUN / 'records').glob('*.json')} == expected_names
    assert read(RUN / 'status.json')['completed_measurements'] == len(records) == 480

    numeric_checks = 0
    def close(actual, expected):
        nonlocal numeric_checks
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
        numeric_checks += 1

    for split, modes in summary['results'].items():
        indices = [j for j, sample in enumerate(samples) if split == 'all' or sample['split'] == split]
        for mode, result in modes.items():
            close(result['actions'], len(indices))
            close(result['mean_windows'], statistics.mean(records[j, 0, mode]['windows'] for j in indices))
            for key in ('coordinate_seconds', 'feature_head_seconds', 'total_seconds'):
                values = sorted(statistics.mean(records[j, r, mode][key] for r in range(2)) for j in indices)
                position = (len(values) - 1) * .95
                lo, hi = math.floor(position), math.ceil(position)
                p95 = values[lo] + (values[hi] - values[lo]) * (position - lo)
                for stat, value in [('mean', statistics.mean(values)), ('median', statistics.median(values)), ('p95', p95)]:
                    close(result[key][stat], value)
            for key in ('peak_allocated_gib', 'peak_reserved_gib'):
                close(result[key], max(records[j, r, mode][key] for j in indices for r in range(2)))
    report = dict(status='passed', run=str(RUN.relative_to(ROOT)), samples=len(samples),
                  measurements=len(records), summary_numeric_checks=numeric_checks,
                  semantic_content_matches=len(samples),
                  source_snapshot=str(SOURCE_ARCHIVE.relative_to(CODE_ROOT)),
                  verified_source_hashes=protocol['source_sha256'],
                  recomputed_cpu_decisions=len(records), cached_coordinate_matches=cached_coordinate_matches,
                  window_counts=dict(Counter(len(s['windows']) for s in samples)),
                  hashes={name: digest(RUN / name) for name in ('protocol.json', 'samples.json', 'summary.json', 'status.json', 'environment.json')},
                  record_hashes={name: digest(RUN / 'records' / name) for name in sorted(expected_names)},
                  current_semantic_crop_hashes=input_hashes,
                  limitations=['No GPU rerun performed by this audit',
                               'Crop/semantic hashes captured at audit time, not benchmark time',
                               'Initial GPU log contains active desktop graphics; no continuous utilization trace',
                               'P95 is over per-action two-repeat means, not individual-call tail latency',
                               'Memory is PyTorch allocator accounting, not whole-device memory'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optional output JSON path')
    args = parser.parse_args()
    report = audit()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if 'hashes' not in k}, ensure_ascii=False, indent=2))
