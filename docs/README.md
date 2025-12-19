# Concept Geometry Pipeline

## Install
```
python -m pip install -e .
```

## Quickstart
1) Capture activations (cached after first run):
```
python -m src.cli capture --concept sentiment --split discovery --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --use_cache 1
```
2) Geometry and controls:
```
python -m src.cli geometry --concept sentiment --split discovery --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --use_cache 1
python -m src.cli controls --concept sentiment --split discovery --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --n_shuffles 50 --use_cache 1
```
3) Ablation:
```
python -m src.cli ablate --concept sentiment --split eval --layer 0 --methods variance,probe --m_list 5,10,20 --model distilgpt2 --adapter gpt2 --config configs/paper.yaml --use_cache 1
```
4) Specificity and behavior:
```
python -m src.cli specificity --concepts sentiment,concreteness --split discovery --model distilgpt2 --adapter gpt2 --use_cache 1
python -m src.cli behavior --concept sentiment --model distilgpt2 --adapter gpt2 --use_cache 1
```
5) End-to-end:
```
python -m src.cli run_all --config configs/paper.yaml --use_cache 1
```

## Adding a concept
- Edit `configs/paper.yaml` (or your config) to add a concept with levels, discovery/eval synonyms, and template families.
- Data generation is keyed by concept name and template family, with disjoint discovery/eval synonyms.

## Adding a model adapter
- Implement `ModelAdapter` subclass in `src/adapters/`, defining `list_layers`, `_capture_batch`, and `_capture_batch_with_ablation`.
- Add to `_select_adapter` in `src/capture.py`.

## Outputs
- Artifacts under `artifacts/` (activations, geometry, controls, ablation, specificity, behavior).
- Raw plot data in `artifacts/raw/`; plots in `artifacts/plots/`; manifests in `artifacts/manifests/`.
- Curated figures in `paper_figures/` with `index.json`.
