# Concept Geometry Toolkit

End-to-end pipeline for multi-concept geometry, controls, ablation, specificity, and behavioral readout with cached activations and manifests.

## Install
```
python -m pip install -e .
```

## Quickstart (end-to-end)
```
python -m src.cli run_all --config configs/paper.yaml --use_cache 1
```
Outputs go to `artifacts/` (plots/raw/stats/manifests) and curated figures to `paper_figures/`.

## Common commands
- Capture: `python -m src.cli capture --concept sentiment --split discovery --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --use_cache 1`
- Geometry: `python -m src.cli geometry --concept sentiment --split discovery --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --use_cache 1`
- Controls: `python -m src.cli controls --concept sentiment --split discovery --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --n_shuffles 50 --use_cache 1`
- Ablation: `python -m src.cli ablate --concept sentiment --split eval --layer 0 --methods variance,probe --m_list 5,10,20 --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --use_cache 1`
- Specificity: `python -m src.cli specificity --concepts sentiment,concreteness --split discovery --model distilgpt2 --adapter gpt2 --use_cache 1`
- Behavior: `python -m src.cli behavior --concept sentiment --model distilgpt2 --adapter gpt2 --use_cache 1`

For more detail, see `docs/README.md`, `docs/EXPERIMENTS.md`, and `docs/ARTIFACTS.md`.
