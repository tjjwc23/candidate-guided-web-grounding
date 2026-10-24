"""Independently verify coverage records using vectorized source-box geometry."""
from collections import defaultdict
import hashlib
import json
from zipfile import ZipFile

import numpy as np

from manifest import ROOT, CODE_ROOT
OUT = ROOT / 'outputs/ch04_submission_controls_20261022/coverage'
read = lambda path: json.loads(path.read_text())
config = read(CODE_ROOT / 'configs/paper.json')
protocol = read(OUT / 'protocol.json')
summary = read(OUT / 'summary.json')
checks = 0
for split, result in summary['splits'].items():
    sources = {k: ROOT / config['frozen_inputs'][k][split] for k in protocol['source_hashes'][split]}
    for kind, path in sources.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol['source_hashes'][split][kind]
    source_rows = {r['action_key']: r for r in read(sources['anchored_rows'])}
    baseline = read(sources['grounding_baseline'])
    records = read(OUT / split / 'records.json')
    flags = defaultdict(dict)
    for record in records:
        row = source_rows[record['action_key']]
        boxes = np.array([b[:4] if b is not None and len(b) >= 4 else [np.nan] * 4
                          for b in row['candidate_bboxes']], dtype=float)
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
        width, height = record['page_size']
        valid_page = valid & (centers >= 0).all(1) & (centers < [width, height]).all(1)
        assert int(valid_page.sum()) == record['valid_page_centers']
        for method, rects in record['layouts'].items():
            assert len(rects) == len(row['windows'])
            windows = np.array(rects, dtype=float)
            assert np.all(windows[:, 0] == 0) and np.all(windows[:, 2] == width)
            assert np.all(windows[:, 3] == min(720, height))
            inside = ((centers[:, None, :] >= windows[None, :, :2]) &
                      (centers[:, None, :] < windows[None, :, :2] + windows[None, :, 2:])).all(2)
            hit = valid & inside.any(1)
            assert np.flatnonzero(hit).tolist() == record['methods'][method]['covered_candidate_ids']
            positive_covered = bool((hit & np.array(row['labels'], dtype=bool)).any())
            assert positive_covered == record['methods'][method]['positive_center_covered']
            flags[method][row['action_key']] = positive_covered
            checks += 1
    scores = {}
    for method, values in flags.items():
        per_task = defaultdict(list)
        for action in baseline:
            per_task[action['annotation_id']].append(values.get(action['sample_id'], False))
        scores[method] = np.array([np.mean(per_task[t]) for t in sorted(per_task)])
        reported = result['methods'][method]
        assert abs(scores[method].mean() * 100 - reported['task_macro_percent']) < 1e-10
        assert sum(values.values()) == reported['covered_actions']
        assert abs(100 * sum(values.values()) / len(baseline) - reported['action_micro_percent']) < 1e-10
        checks += 3
    bootstrap = np.random.default_rng(26092026).integers(len(per_task), size=(10000, len(per_task)), dtype=np.int32)
    for method, diff in result['guided_minus_uniform'].items():
        deltas = 100 * (scores['candidate_guided'] - scores[method])
        assert abs(deltas.mean() - diff['delta_pp']) < 1e-10
        assert np.allclose(np.quantile(deltas[bootstrap].mean(1), [.025, .975]), diff['ci95_pp'], atol=1e-10, rtol=0)
        checks += 3
with ZipFile(CODE_ROOT / 'papers/measurement-sources-20261022.zip') as archive:
    prefix = 'coverage-source-20261022/'
    source, = json.loads(archive.read(prefix + 'provenance.json')).values()
    actual = hashlib.sha256(archive.read(prefix + source['file'])).hexdigest()
    assert actual == source['sha256'] == protocol['script_sha256']
report = dict(status='passed', prepared_actions=sum(s['prepared_actions'] for s in summary['splits'].values()),
              denominator_actions=sum(s['actions'] for s in summary['splits'].values()), checks=checks,
              scope='Independent vectorized geometry from original boxes and labels; all per-action flags, macro/micro totals and paired intervals; source hashes unchanged')
(OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False))
