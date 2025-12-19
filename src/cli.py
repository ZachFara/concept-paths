from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from . import config as cfgmod
from .experiments.pipeline import run_ablation, run_geometry
from .io import ensure_run_dir, save_config_snapshot, write_manifest, write_run_metadata
from .plots import plot_metric_by_layer, plot_with_band
from .utils import ensure_dir


def load_config(path: Path | None) -> cfgmod.RunConfig:
    if path is None:
        return cfgmod.RunConfig()
    with path.open() as f:
        data = yaml.safe_load(f)
    return cfgmod.RunConfig.from_dict(data or {})


def save_report(run_dir: Path, content: str) -> None:
    ensure_dir(run_dir)
    (run_dir / "report.md").write_text(content)


def render_report(run_dir: Path, notes: Dict[str, Any]) -> None:
    lines = ["# Experiment Report", ""]
    for k, v in notes.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("See plots/ for figures and artifacts/ for arrays and manifests.")
    save_report(run_dir, "\n".join(lines))


def write_methods(run_dir: Path) -> None:
    text = """# Methods (auto-generated)
- Stimuli: templated sentences with a single `{w}` slot; discovery vs eval families are disjoint. Optional label permutation and unordered concepts for null controls.
- Delta pairs: adjacent ordinal steps within the same template; strategies: cartesian (all synonym pairs) or random (one pair per edge).
- Activations: last-token pooled residual stream captured per transformer block via nnsight.
- PCA: per-layer PCA on Δh; record PC1 explained variance and k90 (PCs to reach 90% variance). PC1 sign anchored so projection correlates positively with ordinal index.
- Rotation: principal angles between top-k PCA subspaces of adjacent layers (k=min(10,k90) or fixed).
- Random/null baselines: random directions/subspaces; label permutation; unordered concept mode.
- Ablation: neurons selected on discovery only; selectors: lookahead, local_corr, local_ridge; controls: random and anti-selected neurons. Effect measured on eval as change in |projection| along concept direction.
"""
    ensure_dir(run_dir / "docs")
    (run_dir / "docs" / "methods.md").write_text(text)


def write_reproducibility(run_dir: Path, cfg: cfgmod.RunConfig) -> None:
    lines = [
        "# Reproducibility",
        "",
        f"- Model: {cfg.model_name}",
        f"- Seeds: {cfg.seeds}",
        f"- Batch size: {cfg.batch_size}",
        f"- Delta pair strategy: {cfg.geometry.delta_pair_strategy}",
        "- Commands:",
        f"  - python -m src.cli run_all --config {run_dir.parent / 'configs/default.yaml'}",
    ]
    ensure_dir(run_dir / "docs")
    (run_dir / "docs" / "reproducibility.md").write_text("\n".join(lines))


def plot_geometry(run_dir: Path, agg: Dict[str, Dict[str, Any]], label: str) -> None:
    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)
    for split in ["discovery", "eval"]:
        plot_metric_by_layer(
            values_by_split={split: agg[split]["k90_mean"]},
            title=f"k90 by layer ({label}, {split})",
            ylabel="k90",
            outpath=plots_dir / f"k90_{label}_{split}.png",
        )
        plot_with_band(
            x=np.arange(len(agg[split]["top_pc_mean"])),
            mean=agg[split]["top_pc_mean"],
            std=agg[split]["top_pc_std"],
            label=f"{split} {label}",
            title=f"Top PC variance ({label}, {split})",
            ylabel="Explained variance (PC1)",
            outpath=plots_dir / f"top_pc_{label}_{split}.png",
        )
        plot_with_band(
            x=np.arange(len(agg[split]["rotation_mean"])),
            mean=agg[split]["rotation_mean"],
            std=agg[split]["rotation_std"],
            label=f"{split} {label}",
            title=f"Subspace rotation ({label}, {split})",
            ylabel="Rotation (deg)",
            outpath=plots_dir / f"rotation_{label}_{split}.png",
        )


def cmd_run_all(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    run_dir = ensure_run_dir(cfg.artifacts_dir, cfg.run_id)
    save_config_snapshot(run_dir, cfg)
    write_run_metadata(run_dir, cfg, seeds=cfg.seeds)

    # Main geometry
    agg = run_geometry(cfg, permute_labels=False, concept_mode="sentiment", run_dir=run_dir / "geometry")
    plot_geometry(run_dir, agg, "main")

    # Permutation control
    agg_perm = run_geometry(cfg, permute_labels=True, concept_mode="sentiment", run_dir=run_dir / "permute")
    plot_geometry(run_dir, agg_perm, "permute")

    # Unordered control
    agg_unordered = run_geometry(cfg, permute_labels=False, concept_mode="unordered", run_dir=run_dir / "unordered")
    plot_geometry(run_dir, agg_unordered, "unordered")

    # Ablation (on main only) for first seed (or multiple seeds aggregated by caller)
    ablation_dir = run_dir / "ablation"
    ensure_dir(ablation_dir)
    ablation_results = {}
    for seed in cfg.seeds:
        for selector in cfg.ablation.selectors:
            res = run_ablation(cfg, run_dir=ablation_dir, selector=selector, seed=seed)
            ablation_results[f"{selector}_seed{seed}"] = {k: v.tolist() for k, v in res.items()}
    write_manifest(ablation_dir / "ablation_results.json", ablation_results)

    render_report(
        run_dir,
        {
            "run_dir": str(run_dir),
            "seeds": cfg.seeds,
            "model": cfg.model_name,
        },
    )
    write_methods(run_dir)
    write_reproducibility(run_dir, cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Concept-paths research runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_all = sub.add_parser("run_all", help="Run full pipeline (geometry + controls + ablation)")
    run_all.add_argument("--config", type=str, default=None, help="Path to YAML config")
    run_all.set_defaults(func=cmd_run_all)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
