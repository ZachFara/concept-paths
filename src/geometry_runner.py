from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import ControlSpec, DataSpec, ExperimentConfig, load_experiment_config
from .capture import capture_activations
from .data import generate_samples
from .metrics import (
    bootstrap_curves,
    compute_deltas,
    compute_pca_metrics,
    compute_rotation_metrics,
    permutation_null,
    summary_scores,
)
from .plots import plot_curves_with_ci, plot_main_vs_control_overlay, plot_null_histogram
from .utils import ensure_dir, save_json


def run_geometry(
    *,
    cfg: ExperimentConfig,
    data_spec: DataSpec,
    control_spec: ControlSpec,
    artifacts_dir: Path,
    adapter: str,
    model: str,
    batch_size: int,
    use_cache: bool = True,
    thresholds: List[float] = [0.8, 0.9, 0.95],
    n_boot: int = 100,
    early_layers: List[int] = [0, 1, 2],
    late_layers: List[int] = [-3, -2, -1],
) -> dict:
    samples, dataset_sig = generate_samples(cfg, data_spec=data_spec, control=control_spec)
    cache = capture_activations(
        config=cfg,
        data_spec=data_spec,
        control_spec=control_spec,
        adapter_name=adapter,
        model_name=model,
        batch_size=batch_size,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    deltas = compute_deltas(samples, cache.residual)
    pc1, k_curves, bases = compute_pca_metrics(deltas, thresholds=thresholds)
    rot = compute_rotation_metrics(bases, k_mode="min10_k90", k90=k_curves[0.9])

    boot = bootstrap_curves(
        deltas,
        n_boot=n_boot,
        thresholds=thresholds,
        k_mode="min10_k90",
        k_fixed=5,
    )

    out_dir = artifacts_dir / "geometry" / f"{data_spec.concept}__{data_spec.split}"
    plots_dir = artifacts_dir / "plots"
    raw_dir = artifacts_dir / "raw"
    stats_dir = artifacts_dir / "stats"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)
    ensure_dir(raw_dir)
    ensure_dir(stats_dir)

    np.savez_compressed(
        out_dir / "metrics.npz",
        pc1=pc1,
        rotation=rot,
        k80=k_curves[0.8],
        k90=k_curves[0.9],
        k95=k_curves[0.95],
    )
    np.savez_compressed(
        out_dir / "bootstrap.npz",
        **boot,
    )
    x = np.arange(pc1.shape[0])
    x_rot = np.arange(rot.shape[0])
    plot_curves_with_ci(
        x=x,
        mean=boot["pc1_mean"],
        ci_low=boot["pc1_ci_low"],
        ci_high=boot["pc1_ci_high"],
        label="pc1",
        title=f"PC1 variance ({data_spec.concept}, {data_spec.split})",
        ylabel="EVR",
        outpath=plots_dir / f"pc1_{data_spec.concept}_{data_spec.split}.png",
        raw_out=raw_dir / f"pc1_{data_spec.concept}_{data_spec.split}.npz",
    )
    plot_curves_with_ci(
        x=x_rot,
        mean=boot["rotation_mean"],
        ci_low=boot["rotation_ci_low"],
        ci_high=boot["rotation_ci_high"],
        label="rotation",
        title=f"Rotation ({data_spec.concept}, {data_spec.split})",
        ylabel="deg",
        outpath=plots_dir / f"rotation_{data_spec.concept}_{data_spec.split}.png",
        raw_out=raw_dir / f"rotation_{data_spec.concept}_{data_spec.split}.npz",
    )
    plot_curves_with_ci(
        x=x,
        mean=boot["k90_mean"],
        ci_low=boot["k90_ci_low"],
        ci_high=boot["k90_ci_high"],
        label="k90",
        title=f"k90 ({data_spec.concept}, {data_spec.split})",
        ylabel="k",
        outpath=plots_dir / f"k90_{data_spec.concept}_{data_spec.split}.png",
        raw_out=raw_dir / f"k90_{data_spec.concept}_{data_spec.split}.npz",
    )
    k_score, r_score = summary_scores(k_curves[0.9], rot, early_layers=early_layers, late_layers=late_layers)
    save_json(
        stats_dir / f"summary_{data_spec.concept}_{data_spec.split}.json",
        {
            "k90_expansion": k_score,
            "rotation_decay": r_score,
            "dataset_signature": dataset_sig,
            "metadata": cache.metadata,
        },
    )
    return {
        "pc1": pc1,
        "k_curves": k_curves,
        "rotation": rot,
        "boot": boot,
        "dataset_signature": dataset_sig,
    }


def run_controls(
    *,
    cfg: ExperimentConfig,
    data_spec: DataSpec,
    control_spec: ControlSpec,
    artifacts_dir: Path,
    adapter: str,
    model: str,
    batch_size: int,
    n_shuffles: int,
    thresholds: List[float],
    early_layers: List[int],
    late_layers: List[int],
    use_cache: bool = True,
) -> dict:
    samples, dataset_sig = generate_samples(cfg, data_spec=data_spec, control=control_spec)
    cache = capture_activations(
        config=cfg,
        data_spec=data_spec,
        control_spec=control_spec,
        adapter_name=adapter,
        model_name=model,
        batch_size=batch_size,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    deltas = compute_deltas(samples, cache.residual)
    pc1, k_curves, bases = compute_pca_metrics(deltas, thresholds=thresholds)
    rot = compute_rotation_metrics(bases, k_mode="min10_k90", k90=k_curves[0.9])
    real_k, real_r = summary_scores(k_curves[0.9], rot, early_layers=early_layers, late_layers=late_layers)

    null_k, null_r = permutation_null(
        samples,
        cache.residual,
        n_shuffles=n_shuffles,
        thresholds=thresholds,
        early_layers=early_layers,
        late_layers=late_layers,
    )
    p_k = (sum(v >= real_k for v in null_k) + 1) / (len(null_k) + 1)
    p_r = (sum(v >= real_r for v in null_r) + 1) / (len(null_r) + 1)

    stats_dir = artifacts_dir / "stats"
    plots_dir = artifacts_dir / "plots"
    raw_dir = artifacts_dir / "raw"
    ensure_dir(stats_dir)
    ensure_dir(plots_dir)
    ensure_dir(raw_dir)
    save_json(
        stats_dir / f"controls_{data_spec.concept}_{data_spec.split}.json",
        {
            "real": {"k90_expansion": real_k, "rotation_decay": real_r},
            "p_values": {"k90_expansion": p_k, "rotation_decay": p_r},
            "null": {"k90_expansion": null_k, "rotation_decay": null_r},
            "dataset_signature": dataset_sig,
        },
    )
    plot_null_histogram(
        null_values=np.array(null_k),
        real_value=real_k,
        title="Permutation null k90 expansion",
        xlabel="score",
        outpath=plots_dir / f"null_k90_{data_spec.concept}_{data_spec.split}.png",
        raw_out=raw_dir / f"null_k90_{data_spec.concept}_{data_spec.split}.npz",
    )
    plot_null_histogram(
        null_values=np.array(null_r),
        real_value=real_r,
        title="Permutation null rotation decay",
        xlabel="score",
        outpath=plots_dir / f"null_rot_{data_spec.concept}_{data_spec.split}.png",
        raw_out=raw_dir / f"null_rot_{data_spec.concept}_{data_spec.split}.npz",
    )
    return {
        "real": {"k90": real_k, "rotation": real_r},
        "null": {"k90": null_k, "rotation": null_r},
        "p_values": {"k90": p_k, "rotation": p_r},
    }
