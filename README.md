# Candidate-Guided Cross-Window Evidence Aggregation for Web Element Grounding

## Method

The method retrieves DOM candidates and uses their locations to select screenshot windows. Predicted coordinates are mapped back to DOM elements, and evidence for the same element is aggregated across windows. An 18-feature residual scorer selects the final element. Semantic ranking provides a fallback when no candidate is hit.

## Installation

Use Python 3.11. Coordinate inference requires CUDA; cached element scoring and scoring-head training also support CPU.

```bash
git clone https://github.com/tjjwc23/candidate-guided-web-grounding.git
cd candidate-guided-web-grounding
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA platform when running GPU inference.

## Running

### Prepare data and checkpoints

The experiments use [Multimodal Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web), [Fara1.5-4B](https://huggingface.co/microsoft/Fara1.5-4B), and the [MindAct candidate-generation model](https://huggingface.co/osunlp/MindAct_CandidateGeneration_deberta-v3-base). Model and dataset loading uses the local Hugging Face cache.

Prepared evidence, adapted checkpoints, and experiment caches must be supplied separately; they are not included in this repository. Configure their paths in:

- `configs/paper.json`: test evidence and reference predictions.
- `configs/element.json`: scoring-head training inputs and inference weights.
- `configs/models.json`: adapted coordinate model.

See [data formats and setup](docs/REPRODUCTION.md) for the required files. Run commands from the repository root and use a new output directory for each experiment.

### Element inference

With prepared coordinate and semantic evidence:

```bash
bash scripts/run.sh infer grounding \
  --split all --output outputs/element-predictions
```

To generate coordinate evidence from prepared screenshot windows:

```bash
bash scripts/run.sh infer coordinate-evidence \
  --split test_task --output outputs/coordinate-evidence
```

Set the corresponding `grounding_points` path in `configs/paper.json` to the generated `points/` directory before running element inference.

### Train the scoring head

```bash
bash scripts/run.sh experiment element-training \
  --device cpu --output outputs/head-training
```

Use `--device cuda` for GPU training. To evaluate the trained head, pass `--head outputs/head-training/head.pt` to `infer grounding`.

### Control experiments

```bash
bash scripts/run.sh experiment element-controls --output outputs/controls
bash scripts/run.sh experiment element-seeds --output outputs/seeds
bash scripts/run.sh experiment element-features --output outputs/features
bash scripts/run.sh experiment element-mechanisms --output outputs/mechanisms
bash scripts/run.sh experiment element-functional --output outputs/functional

PYTHONPATH=src python experiments/cached_revision_controls.py \
  --functional-results outputs/functional --output outputs/first-window
```

The first-window experiment uses the multiwindow heads and tuned coefficients produced by `element-functional`.

### Coverage and latency

```bash
PYTHONPATH=src python experiments/submission_coverage.py --output outputs/coverage
PYTHONPATH=src python experiments/submission_latency.py --output outputs/latency
```

Coverage compares window layouts at matched budgets. Latency requires CUDA and measures coordinate generation and element scoring with cached candidates, semantic scores, and crops.

For command options, append `--help` to any command.
