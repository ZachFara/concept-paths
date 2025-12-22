# Artifacts

All outputs are stored under the run directory created by `run_all` (e.g., `artifacts/<run_id>/`).

## Directories

- `plots/`: PNGs for all curves and summaries
- `stats/`: NPZ and JSON statistics backing the plots
- `paper_figures/`: curated copy of plots for paper figures
- `index.json`: list of exported paper figures
- `docs/`: auto-generated methods and reproducibility notes

## Common files

- `*_curves.npz` and `*_curves.csv`: raw curves behind each plot
- `*_ci.npz`: bootstrap confidence intervals
- `*_null.npz` and `*_pvalues.json`: permutation control distributions
- `*_summary.json`: ablation summary (AUC, CI, p-values)
- `*_manifest.json`: per-command run manifest (timestamp, model, device, cache keys)

## Manifests

Each command writes a manifest with:
- timestamp, git commit, python version, device
- model id, backend, nnsight version
- dataset signature and split signature
- cache keys (when applicable)
