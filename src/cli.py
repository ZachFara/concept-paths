from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml

from . import config as cfgmod
from .experiments.pipeline import run_ablation as run_ablation_legacy, run_geometry
from .io import ensure_run_dir, save_config_snapshot, write_manifest, write_run_metadata
from .plots import plot_metric_by_layer, plot_with_band
from .plotting import plot_curve_with_ci, plot_null_hist
from .ablation import (
    run_ablation as run_ablation_stage4,
    run_ablation_layer_sweep,
    save_ablation_artifacts,
    plot_ablation_curves,
)
from .utils import ensure_dir, save_json
from .capture import build_or_load_activation_cache, dataset_from_samples, load_model_bundle
from .data import generate_samples
from .metrics import (
    compute_deltas,
    compute_pca_metrics,
    compute_rotation_metrics,
    bootstrap_curves,
)


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


def _resolve_template_family(
    concept: str,
    template_family: str | None,
    project_cfg: cfgmod.ProjectConfig,
) -> str:
    if template_family:
        return template_family
    concept_cfg = project_cfg.data.concepts[concept]
    return sorted(concept_cfg.templates.keys())[0]


def _save_curves_csv(path: Path, *, curves: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    keys = list(curves.keys())
    layers = np.arange(len(next(iter(curves.values()))), dtype=int)
    rows = ["layer," + ",".join(keys)]
    for i in range(len(layers)):
        vals = [str(float(curves[k][i])) for k in keys]
        rows.append(f"{i}," + ",".join(vals))
    path.write_text("\n".join(rows))


def cmd_geometry(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.load_config(Path(args.config)) if args.config else cfgmod.ProjectConfig()
    concept = args.concept
    split = args.split
    template_family = _resolve_template_family(concept, args.template_family, project_cfg)
    samples = generate_samples(
        concept,
        split,
        template_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )
    bundle = load_model_bundle(
        args.model,
        device=torch.device(args.device) if args.device else None,
        local_files_only=bool(args.local_files_only),
    )
    dataset = dataset_from_samples(samples)
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=Path("artifacts"),
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
    )
    residual = cache.residual
    deltas = compute_deltas(samples, residual, method="adjacent")
    pca = compute_pca_metrics(deltas)
    rot = compute_rotation_metrics(pca.subspaces)

    rng = np.random.default_rng(args.seed)
    bands = bootstrap_curves(
        samples,
        residual,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )

    stats_dir = Path("artifacts") / "stats"
    plots_dir = Path("artifacts") / "plots"
    ensure_dir(stats_dir)
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)
    ensure_dir(plots_dir)
    safe_model = args.model.replace("/", "__")
    stem = f"{concept}__{split}__{template_family}__{safe_model}"

    curves = {
        "pc1": pca.pc1_curve,
        "k80": pca.k_curves["k80"].astype(np.float32),
        "k90": pca.k_curves["k90"].astype(np.float32),
        "k95": pca.k_curves["k95"].astype(np.float32),
    }
    np.savez_compressed(stats_dir / f"{stem}_curves.npz", **curves)
    _save_curves_csv(stats_dir / f"{stem}_curves.csv", curves=curves)
    np.savez_compressed(stats_dir / f"{stem}_ci.npz", **bands)
    if rot.rotation_curve.size:
        np.savez_compressed(stats_dir / f"{stem}_rotation.npz", rotation=rot.rotation_curve)
        _save_curves_csv(
            stats_dir / f"{stem}_rotation.csv",
            curves={"rotation": rot.rotation_curve},
        )

    layers = np.arange(len(pca.pc1_curve))
    plot_curve_with_ci(
        x=layers,
        mean=pca.pc1_curve,
        low=bands["pc1_low"],
        high=bands["pc1_high"],
        label="pc1",
        title=f"PC1 variance ({concept}, {split})",
        ylabel="Explained variance",
        outpath=plots_dir / f"{stem}_pc1.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=pca.k_curves["k90"].astype(np.float32),
        low=bands["k90_low"],
        high=bands["k90_high"],
        label="k90",
        title=f"k90 ({concept}, {split})",
        ylabel="k90",
        outpath=plots_dir / f"{stem}_k90.png",
    )
    if rot.rotation_curve.size:
        plot_curve_with_ci(
            x=np.arange(len(rot.rotation_curve)),
            mean=rot.rotation_curve,
            low=bands["rotation_low"],
            high=bands["rotation_high"],
            label="rotation",
            title=f"Rotation ({concept}, {split})",
            ylabel="Rotation (deg)",
            outpath=plots_dir / f"{stem}_rotation.png",
        )


def _permute_samples(
    samples: Sequence[Any],
    *,
    seed: int,
    levels: Sequence[str],
) -> list[Any]:
    rng = np.random.default_rng(seed)
    level_ids = np.array([int(s.metadata.get("level_id", -1)) for s in samples], dtype=int)
    if (level_ids < 0).any():
        raise ValueError("Sample metadata missing level_id")
    perm = level_ids.copy()
    rng.shuffle(perm)
    updated = []
    for s, new_level_id in zip(samples, perm, strict=True):
        metadata = dict(s.metadata)
        metadata["control"] = "random_label"
        metadata["original_level"] = s.level
        metadata["level_id"] = int(new_level_id)
        new_level = levels[int(new_level_id)]
        updated.append(
            type(s)(
                sample_id=s.sample_id,
                concept_name=s.concept_name,
                level=new_level,
                template_id=s.template_id,
                synonym=s.synonym,
                prompt_text=s.prompt_text,
                metadata=metadata,
            )
        )
    return updated


def cmd_controls(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.load_config(Path(args.config)) if args.config else cfgmod.ProjectConfig()
    concept = args.concept
    split = args.split
    template_family = _resolve_template_family(concept, args.template_family, project_cfg)
    samples = generate_samples(
        concept,
        split,
        template_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )
    bundle = load_model_bundle(
        args.model,
        device=torch.device(args.device) if args.device else None,
        local_files_only=bool(args.local_files_only),
    )
    dataset = dataset_from_samples(samples)
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=Path("artifacts"),
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
    )
    residual = cache.residual

    base_deltas = compute_deltas(samples, residual, method="adjacent")
    base_pca = compute_pca_metrics(base_deltas)
    base_rot = compute_rotation_metrics(base_pca.subspaces)
    observed_pc1_mean = float(np.nanmean(base_pca.pc1_curve))
    observed_rot_mean = float(np.nanmean(base_rot.rotation_curve))

    rng = np.random.default_rng(args.seed)
    null_pc1 = np.zeros((args.n_shuffles,), dtype=np.float32)
    null_rot = np.zeros((args.n_shuffles,), dtype=np.float32)
    for i in range(args.n_shuffles):
        perm_samples = _permute_samples(
            samples,
            seed=int(rng.integers(0, 1_000_000)),
            levels=project_cfg.data.concepts[concept].levels,
        )
        deltas = compute_deltas(perm_samples, residual, method="adjacent")
        pca = compute_pca_metrics(deltas)
        rot = compute_rotation_metrics(pca.subspaces)
        null_pc1[i] = float(np.nanmean(pca.pc1_curve))
        null_rot[i] = float(np.nanmean(rot.rotation_curve))

    p_pc1 = float(np.mean(null_pc1 >= observed_pc1_mean))
    p_rot = float(np.mean(null_rot >= observed_rot_mean))

    stats_dir = Path("artifacts") / "stats"
    plots_dir = Path("artifacts") / "plots"
    safe_model = args.model.replace("/", "__")
    stem = f"{concept}__{split}__{template_family}__{safe_model}"
    np.savez_compressed(
        stats_dir / f"{stem}_null.npz",
        null_pc1_mean=null_pc1,
        null_rotation_mean=null_rot,
        observed_pc1_mean=np.array([observed_pc1_mean], dtype=np.float32),
        observed_rotation_mean=np.array([observed_rot_mean], dtype=np.float32),
    )
    save_json(
        stats_dir / f"{stem}_pvalues.json",
        {"pc1_mean": p_pc1, "rotation_mean": p_rot},
    )
    plot_null_hist(
        values=null_pc1,
        observed=observed_pc1_mean,
        title=f"PC1 mean null ({concept}, {split})",
        xlabel="PC1 mean",
        outpath=plots_dir / f"{stem}_null_pc1.png",
    )
    plot_null_hist(
        values=null_rot,
        observed=observed_rot_mean,
        title=f"Rotation mean null ({concept}, {split})",
        xlabel="Rotation mean (deg)",
        outpath=plots_dir / f"{stem}_null_rotation.png",
    )


def cmd_ablate(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.ProjectConfig()
    concept = args.concept
    split = args.split
    template_family = _resolve_template_family(concept, args.template_family, project_cfg)
    samples = generate_samples(
        concept,
        split,
        template_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )
    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    if args.layer < 0:
        sweep = run_ablation_layer_sweep(
            samples,
            model_name=args.model,
            selection_method=args.method,
            m=m_list[0],
            alpha=args.alpha,
            batch_size=args.batch_size,
            seed=args.seed,
            artifacts_dir=Path("artifacts"),
        )
        plots_dir = Path("artifacts") / "plots"
        stats_dir = Path("artifacts") / "stats"
        ensure_dir(plots_dir)
        ensure_dir(stats_dir)
        safe_model = args.model.replace("/", "__")
        stem = f"{concept}__{split}__{template_family}__{safe_model}__Lall__{args.method}"
        np.savez_compressed(stats_dir / f"{stem}_layer_sweep.npz", **sweep)
        plot_curve_with_ci(
            x=np.arange(len(sweep["layer_effects"])),
            mean=sweep["layer_effects"],
            low=None,
            high=None,
            label="effect",
            title="Ablation effect vs layer",
            ylabel="Projection delta",
            outpath=plots_dir / f"{stem}_effect_vs_layer.png",
        )
        return

    result = run_ablation_stage4(
        samples,
        model_name=args.model,
        selection_method=args.method,
        layer=args.layer,
        m_list=m_list,
        alpha=args.alpha,
        random_control=bool(args.random_control),
        batch_size=args.batch_size,
        seed=args.seed,
        artifacts_dir=Path("artifacts"),
    )
    plots_dir = Path("artifacts") / "plots"
    stats_dir = Path("artifacts") / "stats"
    ensure_dir(plots_dir)
    ensure_dir(stats_dir)
    safe_model = args.model.replace("/", "__")
    stem = f"{concept}__{split}__{template_family}__{safe_model}__L{args.layer}__{args.method}"
    save_ablation_artifacts(result, out_dir=stats_dir, stem=stem)
    plot_ablation_curves(result, out_dir=plots_dir, stem=stem)


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
            res = run_ablation_legacy(cfg, run_dir=ablation_dir, selector=selector, seed=seed)
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

    geometry = sub.add_parser("geometry", help="Compute geometry metrics + CI from cached activations")
    geometry.add_argument("--config", type=str, default=None, help="Path to YAML config")
    geometry.add_argument("--concept", type=str, default="sentiment")
    geometry.add_argument("--split", type=str, default="discovery")
    geometry.add_argument("--template_family", type=str, default=None)
    geometry.add_argument("--n_per_level", type=int, default=2)
    geometry.add_argument("--seed", type=int, default=0)
    geometry.add_argument("--model", type=str, default="distilgpt2")
    geometry.add_argument("--batch_size", type=int, default=16)
    geometry.add_argument("--device", type=str, default=None)
    geometry.add_argument("--use_cache", type=int, default=1)
    geometry.add_argument("--n_bootstrap", type=int, default=200)
    geometry.add_argument("--local_files_only", type=int, default=1)
    geometry.set_defaults(func=cmd_geometry)

    controls = sub.add_parser("controls", help="Permutation controls with null distributions")
    controls.add_argument("--config", type=str, default=None, help="Path to YAML config")
    controls.add_argument("--concept", type=str, default="sentiment")
    controls.add_argument("--split", type=str, default="discovery")
    controls.add_argument("--template_family", type=str, default=None)
    controls.add_argument("--n_per_level", type=int, default=2)
    controls.add_argument("--seed", type=int, default=0)
    controls.add_argument("--model", type=str, default="distilgpt2")
    controls.add_argument("--batch_size", type=int, default=16)
    controls.add_argument("--device", type=str, default=None)
    controls.add_argument("--use_cache", type=int, default=1)
    controls.add_argument("--n_shuffles", type=int, default=50)
    controls.add_argument("--local_files_only", type=int, default=1)
    controls.set_defaults(func=cmd_controls)

    ablate = sub.add_parser("ablate", help="Run ablation pipeline")
    ablate.add_argument("--concept", type=str, default="sentiment")
    ablate.add_argument("--split", type=str, default="eval")
    ablate.add_argument("--template_family", type=str, default=None)
    ablate.add_argument("--n_per_level", type=int, default=2)
    ablate.add_argument("--seed", type=int, default=0)
    ablate.add_argument("--model", type=str, default="distilgpt2")
    ablate.add_argument("--batch_size", type=int, default=8)
    ablate.add_argument("--layer", type=int, required=True)
    ablate.add_argument("--method", type=str, default="variance")
    ablate.add_argument("--m_list", type=str, default="5,10,20,40,80")
    ablate.add_argument("--alpha", type=float, default=0.05)
    ablate.add_argument("--random_control", type=int, default=1)
    ablate.set_defaults(func=cmd_ablate)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
