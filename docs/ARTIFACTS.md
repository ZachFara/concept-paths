# Artifacts Reference

- `artifacts/activations/*.npz`: Cached residual/MLP activations with metadata.
- `artifacts/geometry/*/metrics.npz`: pc1/k thresholds/rotation per split/concept.
- `artifacts/geometry/*/bootstrap.npz`: Bootstrap CI arrays.
- `artifacts/stats/*.json`: Summaries (geometry, controls, probe, behavior, ablation).
- `artifacts/plots/*.png`: Figures for geometry, controls, ablation, specificity, behavior.
- `artifacts/raw/*.npz`: Raw data backing plots (curves, histograms).
- `artifacts/ablation/*/dose_*.npz/png`: Dose-response curves per method.
- `artifacts/manifests/*.json`: Run manifests with metadata and signatures.
- `paper_figures/`: Curated subset of plots with `index.json`.
