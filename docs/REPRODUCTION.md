# Data formats and reproduction

## Required artifacts

The configuration retains the relative artifact names used for the paper experiments. The expected layout is:

```text
datasets/element/
  evidence_v1/train/
    inference_inputs.json
    supervision.json
    fit_valid_baseline.json
    protocol.json
    points/<index>.json
    semantics/<index>.json
  evidence_v1/<test split>/semantics/<index>.json
  features/
    train.json
    <test split>.json
    <test split>-single.json
outputs/
  checkpoints/
    coordinate/model.json
    coordinate/adapter/
    element/main/model.json
    element/main/head.pt
  element-reference/main-<split>.json
  mainline/
    grounding/adapter/adapter_model.safetensors
    inputs/<test split>/
      geometry/rows.json
      geometry/baseline.json
      geometry/page_index.jsonl
      points/<index>.json
      proposals.json
resources/models/deberta-v3-base/pytorch_model.bin
```

Test splits are `test_task`, `test_website`, and `test_domain`. Reference head predictions also include `main-valid.json`. The element and coordinate exports include a `model.json` describing architecture, base-model revision, and file hashes. A raw `head.pt` alone is not an inference export.

`configs/paper.json` binds the test inputs and reference predictions to their hashes. `configs/element.json` specifies the head-fitting evidence, features, and inference head. To inspect the configuration or validate supplied inputs:

```bash
bash scripts/run.sh inspect inputs --print
bash scripts/run.sh inspect inputs --verify-inputs
bash scripts/run.sh inspect inputs --verify
```

`--verify-inputs` checks the evidence and reference predictions. `--verify` also checks the configured adapted coordinate weights, candidate model, and element head.

For artifacts stored elsewhere:

```bash
export WEBAGENT_REPO=/path/to/artifacts
bash scripts/run.sh infer grounding --split all --output /path/to/new-results
```

Code and configuration are read from this repository; relative artifact paths resolve under `WEBAGENT_REPO`. Output paths resolve as passed on the command line.

## Input records

| File | Required contents |
|---|---|
| `rows.json` | Action identity (`index`, `action_key`, `task_id`), task text, up to four preceding actions, ordered candidate boxes and labels, selected windows |
| `page_index.jsonl` | Action identity and `[width, height]` page size |
| `baseline.json` | Every evaluation action, including preparation failures; `sample_id` and `annotation_id` define the full action/task denominator |
| `points/<index>.json` | Matching action identity and one record per window: top position, crop size, generated coordinate, token log-probability |
| `semantics/<index>.json` | Matching action identity and ordered `semantic_logits` |
| Fitting `inference_inputs.json` | Prepared actions with task identity and `partition` (`fit` or `valid`) |
| `supervision.json` | Candidate-label arrays keyed by action identity |
| `fit_valid_baseline.json` | Full denominators for fitting/validation tasks |
| Fitting `protocol.json` | `checkpoint_sha256` and `semantic_model_sha256` binding the upstream models |

Candidate boxes use page coordinates `[x, y, width, height]`. Generated coordinates use the normalized `[0, 1000)` crop coordinate system. Each window records its crop path, vertical offset, and candidate-center assignments. Crop paths resolve under the artifact root.

## Paper protocol

The upstream Fara coordinate adapter is fixed during scoring-head experiments. Head fitting and validation use disjoint complete tasks from the official training set: 419 fitting tasks contain 3,006 prepared actions, of which 1,860 provide ranking supervision; 130 tasks are used for validation. These subsets are disjoint from the test tasks. The upstream adapter was trained on the official training tasks before the head split.

The scorer has 18 inputs and hidden layers of width 40. The three fitting seeds are 20261012, 20261013, and 20261014. Learning rate and epoch are selected on validation data. The reported main head uses seed 20261012, learning rate 0.003, and epoch 19. See `configs/element.json` for the complete search settings.

The main evaluation uses all test actions and task-macro EA. Paired intervals use 10,000 task-level bootstrap samples. Training-seed standard deviations and task-bootstrap intervals quantify different variation and are reported separately.

## Experiment mapping

| Paper analysis | Entry point |
|---|---|
| Linear versus nonlinear fusion; fixed first-window intervention | `experiment element-controls` |
| Three-seed head fits | `experiment element-seeds` |
| Two-feature versus 18-feature fits | `experiment element-features` |
| Window coverage groups and feature interventions | `experiment element-mechanisms` |
| Validation-tuned fixed scores; all-hit versus minimum-box fitting | `experiment element-functional` |
| Separately fitted first-window heads; complete C50 semantics | `papers/cached_revision_controls.py --functional-results <functional output>` |
| Matched-budget geometric coverage | `papers/submission_coverage.py --output <new directory>` |
| Paired cached-input latency | `papers/submission_latency.py --output <new directory>` |

The first-window script uses the heads and summaries produced by the functional experiment. Complete C50 semantics may read the pinned raw dataset to recover retrieval-only actions that lack usable screenshot geometry.

## Original measurement verification

The measurement verifiers expect the original records in:

```text
outputs/ch04_submission_controls_20261022/coverage/
outputs/ch04_submission_controls_20261022/latency_v2/
outputs/ch04_element18_20261021/main/head.pt
```

They also read the geometry, semantic evidence, screenshots, and adapter files named in the saved protocols. Run:

```bash
PYTHONPATH=src python papers/verify_submission_coverage.py
PYTHONPATH=src python papers/verify_submission_latency.py --report outputs/latency-check.json
```

These scripts check the paper's specific measurement protocol and original records. They are separate from running new coverage or latency experiments. The source files in `papers/measurement-sources-20261022.zip` are used for hash comparison, not imported or executed. The recorded original source paths remain unchanged inside that archive.
