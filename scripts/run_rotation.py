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
from src.metrics import deltas_from_pairs, pca_metrics_by_layer, rotation_by_layer
from src.plots import plot_metric_by_layer
from src.utils import atomic_save_npz, ensure_dir, get_device, maybe_load_npz, save_json, set_seed


def deltas_cache_path(artifacts_dir: Path, *, split: str, model_name: str, seed: int, strategy: str) -> Path:
    safe_model = model_name.replace("/", "__")
    return artifacts_dir / "deltas" / f"{split}__{safe_model}__seed{seed}__{strategy}.npz"


def rotation_cache_path(
    artifacts_dir: Path, *, split: str, model_name: str, seed: int, strategy: str, k_mode: str, metric: str
) -> Path:
    safe_model = model_name.replace("/", "__")
    return artifacts_dir / "rotation" / f"{split}__{safe_model}__seed{seed}__{strategy}__{k_mode}__{metric}.npz"


def main() -> None:
    cfg = RunConfig()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.model_name)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--delta-pair-strategy", choices=["cartesian", "random"], default=cfg.delta_pair_strategy)
    ap.add_argument("--pca-solver", choices=["full", "randomized"], default=cfg.pca_solver)
    ap.add_argument("--k-mode", choices=["fixed", "min10_k90"], default=cfg.rotation_k_mode)
    ap.add_argument("--k-fixed", type=int, default=cfg.rotation_k_fixed)
    ap.add_argument("--rotation-metric", choices=["mean_deg", "sum_deg"], default=cfg.rotation_metric)
    ap.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=cfg.local_files_only)
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(cfg.artifacts_dir)
    ensure_dir(cfg.plots_dir)

    lm = load_lm(args.model, device=device, local_files_only=args.local_files_only)

    rotation_by_split: dict[str, np.ndarray] = {}

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

        # k90 is only needed for k_mode=min10_k90
        if args.k_mode == "min10_k90":
            k90 = pca_metrics_by_layer(deltas, solver=args.pca_solver).k90
        else:
            k90 = None

        rot = rotation_by_layer(
            deltas,
            solver=args.pca_solver,
            k_mode=args.k_mode,
            k_fixed=args.k_fixed,
            k90=k90,
            metric=args.rotation_metric,
        )
        rpath = rotation_cache_path(
            cfg.artifacts_dir,
            split=split,
            model_name=args.model,
            seed=args.seed,
            strategy=args.delta_pair_strategy,
            k_mode=args.k_mode,
            metric=args.rotation_metric,
        )
        atomic_save_npz(rpath, rotation=rot.astype(np.float32))
        save_json(
            rpath.with_suffix(".json"),
            {
                "split": split,
                "model": args.model,
                "seed": args.seed,
                "delta_pair_strategy": args.delta_pair_strategy,
                "pca_solver": args.pca_solver,
                "k_mode": args.k_mode,
                "k_fixed": args.k_fixed,
                "rotation_metric": args.rotation_metric,
                "n_pairs": int(deltas.shape[0]),
                "n_layers": int(deltas.shape[1]),
                "hidden": int(deltas.shape[2]),
            },
        )

        rotation_by_split[split] = rot

    plot_metric_by_layer(
        values_by_split=rotation_by_split,
        title="PCA subspace rotation between adjacent layers",
        ylabel=f"Rotation ({args.rotation_metric})",
        outpath=cfg.plots_dir / "subspace_rotation_by_layer.png",
    )


if __name__ == "__main__":
    main()
