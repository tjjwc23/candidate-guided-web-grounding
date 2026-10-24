"""Measure target-center coverage under matched window budgets."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from manifest import ROOT, CODE_ROOT
SPLITS = ('test_task', 'test_website', 'test_domain')
METHODS = ('candidate_guided', 'uniform_endpoints', 'uniform_cells')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def layouts(page_size, stored_windows):
    """No labels or positive-element locations enter any layout."""
    width, height = page_size
    h = min(720, height)
    m = len(stored_windows)
    assert width > 0 and height > 0 and 1 <= m <= 3
    last = height - h
    # A single endpoint window is the first viewport, as in linspace(0,last,1).
    endpoints = [round(j * last / (m - 1)) for j in range(m)] if m > 1 else [0]
    cells = [round(max(0, min(last, (j + .5) * height / m - h / 2))) for j in range(m)]
    guided = [w['top'] for w in stored_windows]
    assert all(0 <= top <= last for top in guided)
    return {name: [[0, top, width, h] for top in tops]
            for name, tops in zip(METHODS, (guided, endpoints, cells))}


def center(box):
    if not box or len(box) < 4:
        return None
    x, y, w, h = map(float, box[:4])
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return x + w / 2, y + h / 2


def covered_ids(boxes, windows):
    return [i for i, box in enumerate(boxes)
            if (p := center(box)) is not None and any(
                x <= p[0] < x + w and y <= p[1] < y + h for x, y, w, h in windows)]


def reconstructed_guided(boxes, page_size):
    """Independent reproduction of Top20 greedy anchors and C50 nonempty filtering."""
    candidates = [p for b in boxes[:20] if (p := center(b)) is not None]
    last = max(0, page_size[1] - 720)
    starts = sorted({0, *(max(0, min(int(y - 360), last)) for _, y in candidates)})
    remaining, tops = list(candidates), []
    for _ in range(3):
        if not remaining:
            break
        top = max(starts, key=lambda s: sum(s <= y < s + 720 for _, y in remaining))
        hit = [p for p in remaining if top <= p[1] < top + 720]
        if not hit:
            break
        tops.append(top)
        remaining = [p for p in remaining if p not in hit]
    # Greedy counts coincident candidate centres separately; removing a coordinate
    # removes all coincident anchors, matching the covered candidate identities.
    assigned, kept = set(), []
    for top in tops:
        ids = set(covered_ids(boxes, [[0, top, page_size[0], min(720, page_size[1] - top)]]))
        if ids - assigned:
            kept.append(top)
            assigned |= ids
    return kept


def metrics(baseline, records):
    tasks = sorted({b['annotation_id'] for b in baseline})
    ti = {t: i for i, t in enumerate(tasks)}
    counts = Counter(b['annotation_id'] for b in baseline)
    by_key = {r['action_key']: r for r in records}
    vectors = {}
    reports = {}
    rng = np.random.default_rng(26092026)
    boot = rng.integers(len(tasks), size=(10000, len(tasks)), dtype=np.int32)
    for name in METHODS:
        values = np.zeros(len(tasks))
        n = 0
        for b in baseline:
            r = by_key.get(b['sample_id'])
            yes = bool(r and r['methods'][name]['positive_center_covered'])
            n += yes
            values[ti[b['annotation_id']]] += yes / counts[b['annotation_id']]
        vectors[name] = values
        reports[name] = dict(covered_actions=n, task_macro_percent=float(values.mean() * 100),
                             action_micro_percent=100 * n / len(baseline),
                             mean_valid_c50_center_fraction_prepared=float(np.mean([
                                 r['methods'][name]['covered_candidates'] / r['valid_page_centers']
                                 if r['valid_page_centers'] else 0 for r in records])) * 100)
    differences = {}
    for name in METHODS[1:]:
        delta = 100 * (vectors['candidate_guided'] - vectors[name])
        differences[name] = dict(delta_pp=float(delta.mean()),
                                ci95_pp=np.quantile(delta[boot].mean(1), [.025, .975]).tolist(),
                                newly_covered=sum(r['methods']['candidate_guided']['positive_center_covered']
                                                 and not r['methods'][name]['positive_center_covered'] for r in records),
                                lost_coverage=sum(r['methods'][name]['positive_center_covered']
                                                  and not r['methods']['candidate_guided']['positive_center_covered'] for r in records))
    return dict(actions=len(baseline), tasks=len(tasks), prepared_actions=len(records),
                missing_preparation_counted_uncovered=len(baseline) - len(records),
                windows_histogram=dict(Counter(len(r['layouts']['candidate_guided']) for r in records)),
                mean_windows_prepared=float(np.mean([len(r['layouts']['candidate_guided']) for r in records])),
                methods=reports, guided_minus_uniform=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = CODE_ROOT / 'configs/paper.json'
    config = read(config_path)
    source_paths = {s: {kind: ROOT / config['frozen_inputs'][kind][s]
                       for kind in ('anchored_rows', 'grounding_baseline', 'grounding_page_index')} for s in SPLITS}
    protocol = dict(schema='matched-window-center-coverage-v1',
                    evaluation='C50 contains a positive whose centre lies in at least one window; not coordinate-hit coverage or EA',
                    selection='No training, parameter search, or model selection',
                    budget='Each action uses the saved method window count, full page width and min(720,page_height) height for every method',
                    baselines=dict(uniform_endpoints='round(j*(H-h)/(m-1)); m=1 uses top=0',
                                   uniform_cells='round(clip((j+0.5)*H/m-h/2,0,H-h))'),
                    duplicates='Retain duplicate/overlapping windows so assigned window budget is exactly matched',
                    candidate_metric='Mean within-action covered/valid-in-page C50 centres, prepared actions only; diagnostic',
                    denominator='All frozen baseline actions, unavailable preparations and absent C50 positives counted uncovered',
                    uncertainty='Paired task bootstrap, 10000 resamples, seed 26092026; unadjusted intervals',
                    source_hashes={s: {k: digest(p) for k, p in paths.items()} for s, paths in source_paths.items()},
                    config_sha256=digest(config_path), script_sha256=digest(__file__))
    # Freeze all comparator definitions before evaluating labels.
    write(args.output / 'protocol.json', protocol)
    summary = {}
    for split, paths in source_paths.items():
        rows, baseline = read(paths['anchored_rows']), read(paths['grounding_baseline'])
        pages = {r['action_key']: r['page_size'] for r in map(json.loads, paths['grounding_page_index'].read_text().splitlines())}
        assert len({r['action_key'] for r in rows}) == len(rows)
        assert {r['action_key'] for r in rows} <= {b['sample_id'] for b in baseline}
        records = []
        for row in rows:
            size = pages[row['action_key']]
            rects = layouts(size, row['windows'])
            assert reconstructed_guided(row['candidate_bboxes'], size) == [w['top'] for w in row['windows']], row['action_key']
            ids = {name: covered_ids(row['candidate_bboxes'], windows) for name, windows in rects.items()}
            assigned = {int(i) for w in row['windows'] for i in w['centres']}
            assert set(ids['candidate_guided']) == assigned, row['action_key']
            assert sum(len(w['centres']) for w in row['windows']) == len(assigned)
            assert len(row['labels']) == len(row['candidate_bboxes'])
            positives = {i for i, label in enumerate(row['labels']) if label}
            records.append(dict(action_key=row['action_key'], task_id=row['task_id'], page_size=size,
                                layouts=rects, valid_page_centers=len(covered_ids(row['candidate_bboxes'], [[0, 0, *size]])),
                                methods={name: dict(covered_candidate_ids=values, covered_candidates=len(values),
                                                    positive_center_covered=bool(positives.intersection(values)))
                                         for name, values in ids.items()}))
        summary[split] = metrics(baseline, records)
        write(args.output / split / 'records.json', records)
        print(split, json.dumps(summary[split], ensure_ascii=False), flush=True)
    write(args.output / 'summary.json', dict(status='complete', splits=summary,
                                            checks='Reconstructed all saved layouts and independently matched saved unique centre assignments'))


if __name__ == '__main__':
    main()
