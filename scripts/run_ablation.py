from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.ablate import (
    build_or_load_mlp_act_cache,
    concept_direction_pc1,
    eval_ablation_effect,
    save_neuron_selections_csv,
    select_top_neurons,
)
from src.capture import build_or_load_activation_cache, load_lm
from src.config import RunConfig
from src.data import build_delta_pairs, generate_samples
from src.metrics import deltas_from_pairs
from src.plots import plot_metric_by_layer
from src.utils import atomic_save_npz, ensure_dir, get_device, save_json, set_seed


def main() -> None:
    cfg = RunConfig()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.model_name)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--delta-pair-strategy", choices=["cartesian", "random"], default=cfg.delta_pair_strategy)
    ap.add_argument("--top-m", type=int, default=cfg.ablation_top_m)
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=cfg.local_files_only)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    ensure_dir(cfg.artifacts_dir)
    ensure_dir(cfg.plots_dir)

    lm = load_lm(args.model, device=device, local_files_only=args.local_files_only)

    # Discovery split: used for concept directions and neuron selection (no leakage).
    d_samples = generate_samples("discovery")
    d_keys = [s.key for s in d_samples]
    d_prompts = [s.prompt for s in d_samples]
    d_cache = build_or_load_activation_cache(
        lm,
        keys=d_keys,
        prompts=d_prompts,
        artifacts_dir=cfg.artifacts_dir,
        split="discovery",
        model_name=args.model,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
        force_recompute=args.force_recompute,
    )
    d_pairs = build_delta_pairs(d_samples, strategy=args.delta_pair_strategy, seed=args.seed)
    d_deltas = deltas_from_pairs(keys=d_cache.keys, acts=d_cache.acts, pairs=d_pairs)  # [P, L, H]

    # Eval split: used only for measuring ablation effects.
    e_samples = generate_samples("eval")
    e_keys = [s.key for s in e_samples]
    e_prompts = [s.prompt for s in e_samples]
    e_cache = build_or_load_activation_cache(
        lm,
        keys=e_keys,
        prompts=e_prompts,
        artifacts_dir=cfg.artifacts_dir,
        split="eval",
        model_name=args.model,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
        force_recompute=args.force_recompute,
    )
    e_pairs = build_delta_pairs(e_samples, strategy=args.delta_pair_strategy, seed=args.seed)

    n_layers = int(d_cache.acts.shape[1])
    max_layer = n_layers - 2  # we use layer and layer+1

    effects = np.zeros((n_layers - 1,), dtype=np.float32)
    random_effects = np.zeros((n_layers - 1,), dtype=np.float32)
    selections = []

    for layer in range(0, max_layer + 1):
        concept_dir = concept_direction_pc1(d_deltas[:, layer, :])

        # Capture / cache MLP act per sample at this layer (discovery) for neuron scoring.
        d_mlp_act = build_or_load_mlp_act_cache(
            lm,
            keys=d_keys,
            prompts=d_prompts,
            artifacts_dir=cfg.artifacts_dir,
            split="discovery",
            model_name=args.model,
            seed=args.seed,
            layer=layer,
            batch_size=args.batch_size,
            device=device,
            force_recompute=args.force_recompute,
        )  # [N, d_mlp]
        d_mlp_deltas = deltas_from_pairs(keys=d_keys, acts=d_mlp_act[:, None, :], pairs=d_pairs)[:, 0, :]

        sel = select_top_neurons(
            deltas_resid=d_deltas,
            deltas_resid_next=d_deltas[:, layer + 1, :],
            deltas_mlp=d_mlp_deltas,
            layer=layer,
            top_m=args.top_m,
        )
        selections.append(sel)

        result = eval_ablation_effect(
            keys=e_cache.keys,
            acts=e_cache.acts,
            pairs=e_pairs,
            lm=lm,
            prompts=e_prompts,
            selection=sel,
            concept_dir=concept_dir,
            d_mlp=int(d_mlp_act.shape[1]),
            batch_size=args.batch_size,
            device=device,
            rng=rng,
        )
        effects[layer] = float(result["effect"])
        random_effects[layer] = float(result["random_effect"])

    save_neuron_selections_csv(selections, cfg.artifacts_dir / "selected_neurons.csv")
    atomic_save_npz(
        cfg.artifacts_dir / "ablation" / "effects.npz",
        effect=effects,
        random_effect=random_effects,
    )
    save_json(
        cfg.artifacts_dir / "ablation" / "effects.json",
        {
            "model": args.model,
            "seed": args.seed,
            "delta_pair_strategy": args.delta_pair_strategy,
            "top_m": args.top_m,
            "n_layers": n_layers,
        },
    )

    plot_metric_by_layer(
        values_by_split={"target": effects, "random_control": random_effects},
        title="Ablation effect on concept projection (eval only)",
        ylabel="Baseline − ablated (mean |projection|)",
        outpath=cfg.plots_dir / "ablation_effect.png",
    )


if __name__ == "__main__":
    main()
