"""Reproduce C50 semantics and paired first-window heads.

Use the exported heads and tuned coefficients from experiment element-functional.
"""

import artifacts as artifact_io
import argparse
from pathlib import Path
import statistics

import numpy as np
import torch

import manifest as config
from artifacts import write_json
from evaluation.grounding_metrics import Metrics; from evaluation.grounding_metrics import correctness; from data.element_inputs import evidence_rows; from grounding.evidence import packed_evidence; from grounding.evidence import choose_scores; from grounding.evidence import fixed_scores; from grounding.element_scoring import load_model
from data.element_inputs import TRAIN; from data.element_inputs import inputs; from data.element_inputs import FEATURES, PREDICTIONS; from data.element_inputs import SEEDS
from grounding.element_training import fit_native_head as fit_head
from grounding.element_features import features as features; from grounding.element_scoring import predict as predict



def full_semantic(split, rows, baseline, data, reference):
    pred = {r['action_key']: r['fallback'] for r in data}
    correct = correctness(rows, pred)
    proposals = artifact_io.read_json(config.frozen_input('public_proposals', split, reference))
    missing = set(proposals['actions']) - set(correct)
    assert missing <= {b['sample_id'] for b in baseline}
    recovered = {}
    if missing:
        from data.grounding_inputs import raw_records
        from retrieval.inputs import proposal_indices
        from retrieval.inputs import build_dom_candidate_sample
        columns = ['annotation_id', 'action_uid', 'cleaned_html', 'pos_candidates', 'neg_candidates']
        for raw in raw_records(split, columns):
            key = str(raw['annotation_id']) + '_' + str(raw['action_uid'])
            if key not in missing:
                continue
            raw['previous_actions'] = []
            sample = build_dom_candidate_sample(raw)
            indices = proposal_indices(proposals, key, sample)[:50]
            recovered[key] = bool(sample['labels'][indices[0]])
        assert set(recovered) == missing
    correct.update(recovered)
    return correct, recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--functional-results', type=Path, required=True,
                        help='Output of experiment element-functional, including exported all_<seed> heads')
    args = parser.parse_args()
    out = args.output
    functional = args.functional_results
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    reference = config.load()
    write_json(out/'protocol.json', dict(
        upstream='Frozen coordinates, semantic scores and candidates; no new VLM inference',
        semantics='Argmax over all C50 logits, no point-hit restriction; missing geometry does not exclude text-only decisions',
        single='Keep first greedy window; reconstruct features and priors; refit native 18-feature heads for three seeds',
        paired_multi='Reuse native-18 all-hit heads with identical seed initialization, data and validation recipe',
        selection='Learning rates, epochs and fixed coefficients selected on head validation only',
        seeds=SEEDS, comparison='Task-macro full denominator; three-seed mean task differences bootstrapped 10000 times',
        bounds='Additional within-benchmark controls; no new independent dataset or end-to-end latency measurement',
        implementation_sha256=artifact_io.file_digest(Path(__file__)),
    ))
    _, valid_baseline, _, _, _, _ = inputs('valid', reference)
    multi_models = {s: load_model(functional/f'all_{s}/head.pt') for s in SEEDS}
    original = {r['action_key']: r for r in artifact_io.read_json(FEATURES/'train.json')}
    single_train = []
    for row in artifact_io.read_json(TRAIN/'inference_inputs.json'):
        points = artifact_io.read_json(TRAIN/'points'/f"{row['index']}.json")
        sem = artifact_io.read_json(TRAIN/'semantics'/f"{row['index']}.json")
        logits = sem.get('logits', sem.get('semantic_logits'))
        assert features(row, points, logits) == original[row['action_key']]
        single_train.append(features(dict(row, windows=row['windows'][:1]),
                                     dict(points, windows=points['windows'][:1]), logits))
    write_json(out/'single_training_features.json', single_train)
    labels = artifact_io.read_json(TRAIN/'supervision.json')
    single_models, training = {}, {}
    for seed in SEEDS:
        single_models[seed], training[str(seed)] = fit_head(
            single_train, labels, valid_baseline, seed, out/f'single_{seed}')
        print('Fitted first-window head:', seed, training[str(seed)]['selected'], flush=True)
    tuned_lse = artifact_io.read_json(functional/'summary.json')['coefficients']
    results = {}
    for split in reference['splits']:
        rows, baseline, _, _, pd, sd = inputs(split, reference)
        data = artifact_io.read_json(FEATURES/f"{split}.json")
        single = artifact_io.read_json(FEATURES/f"{split}-single.json")
        evidence, _, _ = evidence_rows(rows, pd, sd, data)
        packed = packed_evidence(evidence)
        metric = Metrics(baseline)
        correct = {}
        records = {}
        for name, scorer, coef in [
            ('fixed_lse', fixed_scores, (1.,1.,1.,2.)),
            ('tuned_lse', fixed_scores, tuned_lse),
        ]:
            p = choose_scores(evidence, scorer(packed, coef))
            correct[name] = correctness(rows, p)
            records[name] = p
        correct['semantic_c50'], recovered = full_semantic(split, rows, baseline, data, reference)
        main_records = artifact_io.read_json(PREDICTIONS/f"main-{split}.json")
        correct['main'] = correctness(rows, {r['action_key']:r['winner'] for r in main_records})
        for seed in SEEDS:
            for name, model, dataset in [('single',single_models[seed],single), ('multi',multi_models[seed],data)]:
                p, rec = predict(model, dataset, 'cpu')
                correct[f'{name}_{seed}'] = correctness(rows,p)
                write_json(out/f'{name}_{seed}'/split/'records.json',rec)
                if name == 'multi':
                    old = artifact_io.read_json(functional/f'all_{seed}'/split/'records.json')
                    assert p == {r['action_key']:r['winner'] for r in old}
        reports = {k:metric.report(v) for k,v in correct.items()}
        for new, old in [('fixed_lse','fixed'),('tuned_lse','tuned_fixed'),('main','main')]:
            expected = artifact_io.read_json(functional/split/'summary.json')['methods'][old]['macro_percent']
            assert abs(reports[new]['macro_percent']-expected)<1e-9
        delta = np.mean([metric.task_scores(metric.vector(correct[f'multi_{s}'])-
                         metric.vector(correct[f'single_{s}']))*100 for s in SEEDS],axis=0)
        paired = dict(mean_delta_pp=float(delta.mean()),
                      ci95_pp=np.quantile(delta[metric.bootstrap].mean(1),[.025,.975]).tolist())
        for name in ('single','multi'):
            values=[reports[f'{name}_{s}']['macro_percent'] for s in SEEDS]
            paired[name]=dict(mean=statistics.mean(values),sd=statistics.stdev(values),values=values)
        results[split]=dict(methods=reports,
            main_minus_semantic=metric.difference(correct['semantic_c50'],correct['main']),
            first_window_retraining=paired, recovered_semantic=recovered)
        write_json(out/split/'correctness.json',correct)
        write_json(out/split/'fixed_predictions.json',records)
        write_json(out/split/'summary.json',results[split])
        print(split, {k:reports[k]['macro_percent'] for k in ['semantic_c50','main']},paired,flush=True)
    write_json(out/'summary.json',dict(status='complete',training=training,splits=results))


if __name__ == '__main__':
    main()
