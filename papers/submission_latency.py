"""Measure paired first-window/multi-window GPU latency from cached inputs."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from manifest import ROOT, CODE_ROOT
SPLITS = ('test_task', 'test_website', 'test_domain')
SEED = 20261022


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, n, repeats):
    config = read(CODE_ROOT / 'configs/paper.json')
    samples, hashes = [], {}
    for split in SPLITS:
        rows_path = ROOT / config['frozen_inputs']['anchored_rows'][split]
        pages_path = ROOT / config['frozen_inputs']['grounding_page_index'][split]
        rows = read(rows_path)
        pages = {r['action_key']: r['page_size'] for r in map(json.loads, pages_path.read_text().splitlines())}
        # No labels, cached correctness, inference time, or model score in selection.
        rows.sort(key=lambda r: hashlib.sha256(f'{SEED}:{split}:{r["action_key"]}'.encode()).hexdigest())
        for row in rows[:n]:
            samples.append(dict(split=split, index=row['index'], action_key=row['action_key'],
                                task_id=row['task_id'], task=row['task'], history=row['history'],
                                candidate_bboxes=row['candidate_bboxes'], windows=row['windows'],
                                page_size=pages[row['action_key']],
                                semantic_path=config['frozen_inputs']['grounding_semantics'][split] + f'/{row["index"]}.json'))
        hashes[split] = dict(rows=digest(rows_path), pages=digest(pages_path))
    protocol = dict(schema='paired-first-multi-latency-v1', seed=SEED,
                    selection=f'First {n} prepared actions per split by SHA256(seed:split:action_key); labels and previous outcomes unused',
                    samples=len(samples), repeats=repeats, batch_size=1, dtype='bfloat16', attention='sdpa',
                    comparison='First saved greedy window versus all saved windows, sequential generation; same main element head on CPU',
                    included='PNG load/decode, prompt/image preprocessing, transfer, coordinate prefill and decoding, feature construction, CPU head and argmax',
                    excluded='Model loading, warmup, initial candidate retrieval, semantic inference, page screenshot acquisition, window crop creation, disk result writes',
                    reported_scope='Given cached candidates, semantic logits and crop files; not full end-to-end agent latency',
                    warmup='Three fixed selected actions with largest page width; multi-window passes before timed work',
                    ordering='Alternate single/multi order by sample index and repeat; two repeats by default',
                    timing='CUDA synchronize at stage boundaries; perf_counter; no per-example empty_cache',
                    memory='Reset peak before each condition; report peak allocated and reserved including resident coordinate model',
                    summary='Per-action average across repeats, then mean/median/p95; global GPU peak; split-balanced sample, not task-macro population estimate',
                    sources=hashes,
                    adapter_sha256=digest(ROOT / 'outputs/mainline/grounding/adapter/adapter_model.safetensors'),
                    head_sha256=digest(ROOT / 'outputs/ch04_element18_20261021/main/head.pt'),
                    source_sha256={str(p.relative_to(CODE_ROOT)): digest(p) for p in [
                        Path(__file__), CODE_ROOT / 'src/grounding/encoding.py',
                        CODE_ROOT / 'src/grounding/coordinates.py', CODE_ROOT / 'src/fara.py',
                        CODE_ROOT / 'src/grounding/element_features.py']})
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('samples.json', samples), ('protocol.json', protocol)]:
        path = output / name
        if path.exists():
            assert read(path) == value, 'Protocol/sample changed; use a new output directory'
        else:
            write(path, value)
    return samples, protocol


def run(output, samples, protocol):
    import torch
    try:
        smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
        smi_info = dict(returncode=smi.returncode, stdout=smi.stdout, stderr=smi.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        smi_info = dict(error=str(e))
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda_build=torch.version.cuda,
               cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count(), nvidia_smi=smi_info)
    missing = [str(ROOT / w['crop']) for r in samples for w in r['windows'] if not (ROOT / w['crop']).is_file()]
    missing += [str(ROOT / r['semantic_path']) for r in samples if not (ROOT / r['semantic_path']).is_file()]
    if not env['cuda_available'] or not env['device_count'] or missing:
        result = dict(status='blocked', reason='CUDA unavailable' if not env['cuda_available'] else 'Missing inputs',
                      environment=env, missing_files=missing, sampled_actions=len(samples),
                      completed_measurements=0, timings=None,
                      resume_command='PYTHONPATH=src python papers/submission_latency.py')
        write(output / 'status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 2

    from fara import load_model
    from grounding.coordinates import coordinate_suffix_tokens
    from grounding.coordinates import greedy_coordinate
    from grounding.element_scoring import ElementScorer as JointReranker; from grounding.element_features import features as features; from grounding.element_scoring import predict as predict
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    def load_latency_state():
        processor, model = load_model()
        set_peft_model_state_dict(
            model, load_file(ROOT / 'outputs/mainline/grounding/adapter/adapter_model.safetensors')
        )
        model.eval()
        return dict(processor=processor, model=model,
                    suffix=coordinate_suffix_tokens(processor.tokenizer))

    torch.set_num_threads(4)
    env.update(gpu=torch.cuda.get_device_name(0),
               total_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
               cpu_threads=torch.get_num_threads())
    # A resumed benchmark must not silently combine different GPU/software states.
    stable_env = {k: v for k, v in env.items() if k != 'nvidia_smi'}
    environment_path = output / 'environment.json'
    if environment_path.exists():
        assert read(environment_path)['stable'] == stable_env, 'Environment changed; use a new output directory'
    else:
        write(environment_path, dict(stable=stable_env, initial_nvidia_smi=smi_info))
    head = JointReranker().eval()
    head.load_state_dict(torch.load(ROOT / 'outputs/ch04_element18_20261021/main/head.pt', map_location='cpu', weights_only=True))
    for row in samples:
        sem = read(ROOT / row['semantic_path'])
        assert sem['action_key'] == row['action_key']
        row['logits'] = sem.get('semantic_logits', sem.get('logits'))

    def measure(state, row, mode):
        selected = dict(row, windows=row['windows'][:1] if mode == 'single' else row['windows'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        points = []
        for window in selected['windows']:
            value = greedy_coordinate(state['processor'], state['model'],
                                      dict(task=row['task'], history=row['history'], crop=window['crop']), state['suffix'])
            points.append(dict(value, top=window['top'],
                               crop_size=[row['page_size'][0], min(720, row['page_size'][1] - window['top'])]))
        torch.cuda.synchronize()
        coordinate_finished = time.perf_counter()
        feat = features(selected, dict(action_key=row['action_key'], windows=points), row['logits'])
        decisions, _ = predict(head, [feat], 'cpu')
        ended = time.perf_counter()
        return dict(action_key=row['action_key'], split=row['split'], mode=mode, windows=len(points),
                    coordinate_seconds=coordinate_finished - started,
                    feature_head_seconds=ended - coordinate_finished, total_seconds=ended - started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                    winner=decisions[row['action_key']], predictions=points)

    lock_path = ROOT / 'outputs/.gpu0.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            write(output / 'status.json', dict(status='blocked', reason='Project GPU lock held by another job', timings=None))
            return 2
        state = load_latency_state()
        warmup = sorted(samples, key=lambda r: (-r['page_size'][0], r['action_key']))[:3]
        for row in warmup:
            measure(state, row, 'multi')
        completed = 0
        for repeat in range(protocol['repeats']):
            for j, row in enumerate(samples):
                modes = ('single', 'multi') if (repeat + j) % 2 == 0 else ('multi', 'single')
                for mode in modes:
                    path = output / 'records' / f'{j:04d}_{repeat}_{mode}.json'
                    if path.exists():
                        assert read(path)['action_key'] == row['action_key']
                        continue
                    record = dict(measure(state, row, mode), repeat=repeat, sample_index=j)
                    write(path, record)
                    completed += 1
                    if completed % 10 == 0:
                        status = dict(status='running', new_completed=completed, repeat=repeat, sample_index=j,
                                      total_planned=len(samples) * protocol['repeats'] * 2)
                        write(output / 'status.json', status)
                        print(json.dumps(status), flush=True)
        del state

    import numpy as np
    summaries = {}
    for split in ('all', *SPLITS):
        summaries[split] = {}
        for mode in ('single', 'multi'):
            rows = [[read(output / 'records' / f'{j:04d}_{r}_{mode}.json') for r in range(protocol['repeats'])]
                    for j, row in enumerate(samples) if split == 'all' or row['split'] == split]
            result = dict(actions=len(rows), mean_windows=float(np.mean([rs[0]['windows'] for rs in rows])))
            for key in ('coordinate_seconds', 'feature_head_seconds', 'total_seconds'):
                times = np.array([np.mean([r[key] for r in rs]) for rs in rows])
                result[key] = dict(mean=float(times.mean()), median=float(np.median(times)), p95=float(np.quantile(times, .95)))
            for key in ('peak_allocated_gib', 'peak_reserved_gib'):
                result[key] = max(r[key] for rs in rows for r in rs)
            summaries[split][mode] = result
    write(output / 'summary.json', dict(status='complete', results=summaries, environment=stable_env))
    write(output / 'status.json', dict(status='complete', completed_measurements=len(samples) * protocol['repeats'] * 2))
    print(json.dumps(summaries, ensure_ascii=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-split', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert args.per_split > 0 and args.repeats > 0
    samples, protocol = prepare(args.output, args.per_split, args.repeats)
    if args.prepare_only:
        print(json.dumps(dict(status='prepared', samples=len(samples), window_counts=dict(Counter(len(r['windows']) for r in samples)))))
        return
    try:
        code = run(args.output, samples, protocol)
    except Exception as exc:
        write(args.output / 'status.json', dict(status='failed', reason=f'{type(exc).__name__}: {exc}'))
        raise
    raise SystemExit(code)


if __name__ == '__main__':
    main()
