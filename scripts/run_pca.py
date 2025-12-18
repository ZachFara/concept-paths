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

from src.capture import build_or_load_activation_cache, load_lm
from src.config import RunConfig
from src.data import build_delta_pairs, generate_samples
from src.metrics import deltas_from_pairs, pca_metrics_by_layer
from src.plots import plot_metric_by_layer
from src.utils import atomic_save_npz, ensure_dir, get_device, maybe_load_npz, save_json, set_seed


def deltas_cache_path(artifacts_dir: Path, *, split: str, model_name: str, seed: int, strategy: str) -> Path:
    safe_model = model_name.replace("/", "__")
    return artifacts_dir / "deltas" / f"{split}__{safe_model}__seed{seed}__{strategy}.npz"


def pca_cache_path(artifacts_dir: Path, *, split: str, model_name: str, seed: int, strategy: str) -> Path:
    safe_model = model_name.replace("/", "__")
    return artifacts_dir / "pca" / f"{split}__{safe_model}__seed{seed}__{strategy}.npz"


def main() -> None:
    cfg = RunConfig()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.model_name)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--delta-pair-strategy", choices=["cartesian", "random"], default=cfg.delta_pair_strategy)
    ap.add_argument("--pca-solver", choices=["full", "randomized"], default=cfg.pca_solver)
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=cfg.local_files_only)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(cfg.artifacts_dir)
    ensure_dir(cfg.plots_dir)

    lm = load_lm(args.model, device=device, local_files_only=args.local_files_only)

    top_pc_by_split: dict[str, np.ndarray] = {}
    k90_by_split: dict[str, np.ndarray] = {}

    for split in ["discovery", "eval"]:
        samples = generate_samples(split)
        keys = [s.key for s in samples]
        prompts = [s.prompt for s in samples]

        cache = build_or_load_activation_cache(
            lm,
            keys=keys,
            prompts=prompts,
            artifacts_dir=cfg.artifacts_dir,
            split=split,
            model_name=args.model,
            seed=args.seed,
            batch_size=args.batch_size,
            device=device,
            force_recompute=args.force_recompute,
        )

        pairs = build_delta_pairs(samples, strategy=args.delta_pair_strategy, seed=args.seed)
        dpath = deltas_cache_path(
            cfg.artifacts_dir,
            split=split,
            model_name=args.model,
            seed=args.seed,
            strategy=args.delta_pair_strategy,
        )
        loaded = None if args.force_recompute else maybe_load_npz(dpath)
        if loaded is not None:
            deltas = loaded["deltas"].astype(np.float32)
        else:
            deltas = deltas_from_pairs(keys=cache.keys, acts=cache.acts, pairs=pairs)
            atomic_save_npz(dpath, deltas=deltas.astype(np.float32))

        pmetrics = pca_metrics_by_layer(deltas, solver=args.pca_solver)
        ppath = pca_cache_path(
            cfg.artifacts_dir,
            split=split,
            model_name=args.model,
            seed=args.seed,
            strategy=args.delta_pair_strategy,
        )
        atomic_save_npz(
            ppath,
            top_pc_ratio=pmetrics.top_pc_ratio.astype(np.float32),
            k90=pmetrics.k90.astype(np.int64),
        )
        save_json(
            ppath.with_suffix(".json"),
            {
                "split": split,
                "model": args.model,
                "seed": args.seed,
                "delta_pair_strategy": args.delta_pair_strategy,
                "pca_solver": args.pca_solver,
                "n_pairs": int(deltas.shape[0]),
                "n_layers": int(deltas.shape[1]),
                "hidden": int(deltas.shape[2]),
            },
        )

        top_pc_by_split[split] = pmetrics.top_pc_ratio
        k90_by_split[split] = pmetrics.k90.astype(np.float32)

    plot_metric_by_layer(
        values_by_split=k90_by_split,
        title="k90 by layer (Δ sentiment steps)",
        ylabel="k90 (PCs to explain 90% variance)",
        outpath=cfg.plots_dir / "pca_k90_by_layer.png",
    )
    plot_metric_by_layer(
        values_by_split=top_pc_by_split,
        title="Top PC explained variance by layer (Δ sentiment steps)",
        ylabel="Explained variance ratio (PC1)",
        outpath=cfg.plots_dir / "top_pc_variance_by_layer.png",
    )


if __name__ == "__main__":
    main()
