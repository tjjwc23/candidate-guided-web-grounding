# Candidate-Guided Cross-Window Evidence Aggregation for Web Element Grounding

Code for **Candidate-Guided Cross-Window Evidence Aggregation for Web Element Grounding**.

The method retrieves DOM candidates, uses their locations to select screenshot windows, and maps predicted coordinates back to DOM elements. Evidence for the same element is aggregated across windows. An 18-feature residual scorer selects the final element, with semantic ranking as the fallback when no candidate is hit.

## Results

Task-macro element accuracy on Multimodal Mind2Web:

| Method | Cross-task | Cross-website | Cross-domain |
|---|---:|---:|---:|
| Complete C50 semantic ranking | 33.71 | 29.26 | 29.45 |
| Original fixed scoring | 61.16 | 54.93 | 49.99 |
| Validation-tuned fixed scoring | 60.73 | 54.21 | 50.41 |
| Proposed method | **63.53** | **55.95** | **52.84** |

Values are percentages. Evaluation includes all 1,339 / 1,019 / 4,060 actions in the three splits; actions without prepared visual evidence count as incorrect. Tasks receive equal weight.

## Installation

Use Python 3.11 and a PyTorch build compatible with your CUDA platform.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run.sh --help
```

The dependency pins record the research environment. Installation on a fresh machine has not yet been verified. Coordinate inference requires CUDA; cached element scoring, head control experiments, and statistical analysis can run on CPU.

## Data and checkpoints

This repository contains code and configuration. Datasets, adapted checkpoints, and experiment caches are not bundled, and download links for the adapted checkpoints and caches are not currently provided.

The upstream resources are:

- [Multimodal Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web), revision `1b4c6a8cf9f77b7a5e0d641959935c80c4a05889`.
- [Fara1.5-4B](https://huggingface.co/microsoft/Fara1.5-4B), revision `776a33ae5b2ad503796a97ae20fdc66f61d2feea`.
- [MindAct candidate-generation model](https://huggingface.co/osunlp/MindAct_CandidateGeneration_deberta-v3-base), revision `92d3ddcb079b1749015d72293c82d640b0b9a1da`.

Download the pinned resources into the Hugging Face cache before using model or raw-data workflows, which load locally cached files. Upstream access and license terms apply.

Configuration files:

| File | Purpose |
|---|---|
| `configs/paper.json` | Test evidence, reference predictions, model hashes, split sizes, and reported EA |
| `configs/element.json` | Head fitting inputs, seeds, hyperparameters, and inference head |
| `configs/models.json` | Adapted coordinate-model export |

Paths are relative to the repository root. Set `WEBAGENT_REPO` to use an external artifact root with the same layout. See [data formats and reproduction](docs/REPRODUCTION.md) for the required files.

## Inference

With the coordinate evidence, semantic scores, and exported element head in place:

```bash
bash scripts/run.sh infer grounding \
  --split all --output outputs/element-predictions
```

The output contains per-action element predictions and full-denominator metrics. The code was checked against the original artifacts: all 6,059 prepared test-action predictions matched the reference results.

To generate coordinate evidence from prepared windows and the adapted coordinate model:

```bash
bash scripts/run.sh infer coordinate-evidence \
  --split test_task --output outputs/coordinate-evidence
```

Update the `grounding_points` paths in `configs/paper.json` to evaluate newly generated evidence.

## Scoring-head training and experiments

Train and select an 18-feature head from prepared fitting/validation evidence:

```bash
bash scripts/run.sh experiment element-training \
  --device cpu --output outputs/head-training
```

The reported main head was fitted on GPU; CPU refitting can produce different predictions. Use the original head for replaying the reported main result.

Run the paper's controls with shared upstream evidence:

```bash
bash scripts/run.sh experiment element-controls --output outputs/controls
bash scripts/run.sh experiment element-seeds --output outputs/seeds
bash scripts/run.sh experiment element-features --output outputs/features
bash scripts/run.sh experiment element-mechanisms --output outputs/mechanisms
bash scripts/run.sh experiment element-functional --output outputs/functional

PYTHONPATH=src python papers/cached_revision_controls.py \
  --functional-results outputs/functional --output outputs/first-window

bash scripts/run.sh verify element-functional --output outputs/functional
```

The first-window experiment consumes the exported multiwindow heads and validation-tuned coefficients produced by `element-functional`. All experiments require the input artifacts described in the reproduction guide. Use a new output directory for each run.

The release covers scoring-head training, coordinate inference, element aggregation, and the paper's control analyses. It does not include training the upstream coordinate adapter from scratch.

## Coverage and latency

```bash
PYTHONPATH=src python papers/submission_coverage.py --output outputs/coverage
PYTHONPATH=src python papers/submission_latency.py --output outputs/latency
```

Coverage compares candidate-guided, endpoint-spaced, and segment-centered windows at matched budgets. Latency measures coordinate generation and element scoring given cached candidates, semantic scores, and crops; it excludes retrieval and initial page preparation.

The `verify_submission_*` scripts check the original saved measurement records. Their source archive is retained to verify the recorded implementation hashes; see the reproduction guide for the required layout.

## Repository layout

```text
configs/          Paper inputs and head-training settings
src/data/         DOM candidates, screenshots, and evidence preparation
src/retrieval/    Candidate ranking
src/grounding/    Coordinate inference, feature aggregation, and scoring heads
src/experiments/  Head fitting and control experiments
src/evaluation/   Metrics, paired statistics, and independent checks
papers/           First-window, coverage, and latency experiments
scripts/run.sh    Command-line entry point
tests/           Unit and artifact-dependent tests
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The default suite uses synthetic or temporary inputs. With the original coordinate exports available, run the artifact-dependent check separately:

```bash
python -m pytest -q -m integration
```
