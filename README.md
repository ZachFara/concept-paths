# Concept-Paths (nnsight)

Minimal research codebase for **concept-path tracking** of sentiment in a causal transformer using **nnsight**.
Upgraded to include multi-seed aggregation, permutation/null controls, random baselines, leakage guards, reproducibility manifests, and unified CLI entrypoints.

It:
- builds controlled template-based sentiment stimuli (discovery vs evaluation are disjoint; multiple template families; optional permutation/unordered nulls),
- captures **residual stream activations** at every transformer block (last-token pooling),
- computes per-layer PCA on small sentiment-step Δ vectors (PC1 variance + k90),
- computes **subspace rotation** between adjacent layers via principal angles,
- runs MLP neuron ablation experiments with multiple selectors and sham controls.

## Setup

Create an environment (conda/venv) with Python ≥ 3.10, then install:

```bash
python -m pip install -e .
```

Notes:
- Default model is `gpt2`. The scripts default to `--local-files-only`, so you need the model cached locally.
- If you need to download models, set `--no-local-files-only` (requires network) or pre-download via Hugging Face.

## Run (end-to-end, unified CLI)

```bash
python -m src.cli run_all --config configs/paper.yaml --backend nnsight --use_cache 1
```

This runs geometry + controls, ablation, specificity/transfer, and behavior probes for sentiment and concreteness, and writes:
- plots in `<artifacts>/<run_id>/plots`
- stats/NPZ/JSON in `<artifacts>/<run_id>/stats`
- paper figure exports in `<artifacts>/<run_id>/paper_figures`
- docs in `<artifacts>/<run_id>/docs/`
- a short `report.md` + `index.json`

## Legacy scripts (shims)
`scripts/run_pca.py`, `scripts/run_rotation.py`, `scripts/run_ablation.py` now forward to the unified CLI with a deprecation notice.

## What to edit

- Dataset templates / adjectives + split rules: `src/config.py`
- Stimulus generation and leakage guards: `src/stimuli.py`
- nnsight capture utilities: `src/capture.py` (newer: `src/activations.py`)
- PCA + rotation metrics: `src/metrics.py` (newer: `src/geometry.py`)
- Ablation experiment: `src/ablate.py` (with multiple selectors/controls)
- Unified orchestration + CLI: `src/experiments/pipeline.py`, `src/cli.py`

## Artifacts and caching

All runs cache intermediate results to avoid recomputation:
- `artifacts/activations/` pooled residuals `[n_samples, n_layers, hidden]`
- `artifacts/deltas/` Δ vectors for adjacent sentiment steps
- `artifacts/pca/` per-layer PCA summaries
- `artifacts/rotation/` rotation metrics
- `artifacts/mlp_act/` (ablation) cached MLP activations

Plots are written to `plots/`.

## Device support

The code will use **Apple Silicon MPS** if available, otherwise CPU.
If nnsight scanning is unstable on MPS for a particular setup, it automatically falls back to CPU for tracing.

See `docs/EXPERIMENTS.md` and `docs/ARTIFACTS.md` for experiment and artifact details.
