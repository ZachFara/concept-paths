# Experiments

## Quickstart

```bash
python -m src.cli run_all --config configs/paper.yaml --backend nnsight --use_cache 1
```

This runs:
- geometry + controls (random labels)
- ablation with dose response
- specificity + transfer
- behavior probes + ablation impact
- for sentiment and concreteness, and for GPT-2 + OPT-125m

By default, the full pipeline runs on the first model only; secondary models run core plots
(geometry + controls). Set `run_all_full_all_models: true` in `configs/paper.yaml` to run
the full pipeline on every model.

## Component commands

```bash
python -m src.cli geometry --concept sentiment --split discovery --use_cache 1
python -m src.cli controls --concept sentiment --split discovery --n_shuffles 50 --use_cache 1
python -m src.cli ablate --concept sentiment --split eval --layer 2 --method variance --m_list 5,10
python -m src.cli specificity --concepts sentiment,concreteness --use_cache 1
python -m src.cli behavior --concept sentiment --use_cache 1
```

## Notes

- nnsight is required for capture/ablation. No hooks backend is supported.
- On Apple Silicon (MPS), nnsight scan can be brittle. The code falls back to CPU for tracing if scan fails.
