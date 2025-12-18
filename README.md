# Concept-Paths (nnsight)

Minimal research codebase for **concept-path tracking** of sentiment in a causal transformer using **nnsight**.

It:
- builds a controlled template-based sentiment sweep dataset (discovery vs evaluation are disjoint),
- captures **residual stream activations** at every transformer block (last-token pooling),
- computes per-layer PCA on small sentiment-step Δ vectors (PC1 variance + k90),
- computes **subspace rotation** between adjacent layers via principal angles,
- (optional) runs a simple **MLP neuron ablation** experiment without label leakage.

## Setup

Create an environment (conda/venv) with Python ≥ 3.10, then install:

```bash
python -m pip install -e .
```

Notes:
- Default model is `gpt2`. The scripts default to `--local-files-only`, so you need the model cached locally.
- If you need to download models, set `--no-local-files-only` (requires network) or pre-download via Hugging Face.

## Run (end-to-end)

From the repo root:

### PCA (k90 + top PC variance)

```bash
python scripts/run_pca.py --model gpt2 --batch-size 16 --seed 0
```

Outputs:
- `plots/pca_k90_by_layer.png`
- `plots/top_pc_variance_by_layer.png`

### Subspace rotation (principal angles)

```bash
python scripts/run_rotation.py --model gpt2 --batch-size 16 --seed 0
```

Outputs:
- `plots/subspace_rotation_by_layer.png`

### Optional: ablation experiment (no leakage)

This:
- uses the **discovery** split to compute a per-layer concept direction (PC1 of Δh_l),
- selects neurons by correlation on discovery only,
- evaluates projection change on **evaluation** only,
- includes random-neuron ablation as a control.

```bash
python scripts/run_ablation.py --model gpt2 --batch-size 16 --seed 0 --top-m 50
```

Outputs:
- `plots/ablation_effect.png`
- `artifacts/selected_neurons.csv`

## What to edit

- Dataset templates / adjectives + split rules: `src/config.py`
- Dataset generation + Δ pair construction: `src/data.py`
- nnsight capture utilities: `src/capture.py`
- PCA + rotation metrics: `src/metrics.py`
- Ablation experiment: `src/ablate.py`

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

