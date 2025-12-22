from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm

from . import config as cfgmod
from .experiments.pipeline import run_ablation as run_ablation_legacy, run_geometry
from .io import ensure_run_dir, save_config_snapshot, write_manifest, write_run_metadata
from .plots import plot_metric_by_layer, plot_with_band
from .plotting import plot_curve_with_ci, plot_null_hist, plot_heatmap, plot_bar
from .stats import empirical_p_stats
from .specificity import (
    similarity_across_concepts,
    permutation_test_similarity,
    transfer_direction,
    shared_variance_control,
    pc1_directions_from_samples,
    cross_ablation_transfer,
)
from .behavior import train_ridge_probes, eval_ridge_probes, ablation_probe_impact
from .ablation import (
    run_ablation as run_ablation_stage4,
    run_ablation_layer_sweep,
    save_ablation_artifacts,
    plot_ablation_curves,
)
from .utils import ensure_dir, save_json, hash_samples, get_device
from .io import git_commit_hash, package_versions
from .capture import build_or_load_activation_cache, dataset_from_samples, load_model_bundle
from .data import generate_samples, generate_samples_all_families
from .metrics import (
    compute_deltas,
    compute_pca_metrics,
    compute_rotation_metrics,
    bootstrap_curves,
    pc1_directions_from_deltas,
    angle_between_directions,
    explained_fraction_along_direction,
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


def _get_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    artifacts_dir = Path(getattr(args, "artifacts_dir", "artifacts"))
    stats_dir = artifacts_dir / "stats"
    plots_dir = artifacts_dir / "plots"
    ensure_dir(stats_dir)
    ensure_dir(plots_dir)
    return artifacts_dir, stats_dir, plots_dir


def _write_manifest(
    *,
    out_dir: Path,
    name: str,
    model_name: str,
    device: str,
    dataset_signature: str | None,
    split_signature: str | None,
    cache_key: dict | None,
    extra: dict | None = None,
) -> None:
    try:
        import nnsight

        nnsight_version = getattr(nnsight, "__version__", "unknown")
    except Exception:
        nnsight_version = "not_installed"
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "python": sys.version,
        "device": device,
        "model_id": model_name,
        "backend": "nnsight",
        "nnsight_version": nnsight_version,
        "dataset_signature": dataset_signature,
        "split_signature": split_signature,
        "cache_key": cache_key,
        "packages": package_versions(),
    }
    if extra:
        manifest.update(extra)
    save_json(out_dir / f"{name}_manifest.json", manifest)


def _export_paper_figures(run_dir: Path) -> None:
    import shutil

    plots_dir = run_dir / "plots"
    out_dir = run_dir / "paper_figures"
    ensure_dir(out_dir)
    figures = []
    if plots_dir.exists():
        for path in sorted(plots_dir.glob("*.png")):
            dest = out_dir / path.name
            shutil.copy2(path, dest)
            figures.append(str(dest.relative_to(run_dir)))
    stats_files = []
    stats_dir = run_dir / "stats"
    if stats_dir.exists():
        for path in sorted(stats_dir.glob("*")):
            stats_files.append(str(path.relative_to(run_dir)))
    plot_files = []
    if plots_dir.exists():
        for path in sorted(plots_dir.glob("*.png")):
            plot_files.append(str(path.relative_to(run_dir)))
    index = {"paper_figures": figures, "plots": plot_files, "stats": stats_files}
    save_json(run_dir / "index.json", index)


def cmd_geometry(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.load_config(Path(args.config)) if args.config else cfgmod.ProjectConfig()
    concept = args.concept
    split = args.split
    template_family = _resolve_template_family(concept, args.template_family, project_cfg)
    if template_family == "aggregated-templates":
        samples = generate_samples_all_families(
            concept,
            split,
            seed=args.seed,
            n_per_level=args.n_per_level,
            data_spec=project_cfg.data,
            control_spec=project_cfg.controls,
            concept_mode=args.concept_mode,
            aggregated_family_name=template_family,
        )
    else:
        samples = generate_samples(
            concept,
            split,
            template_family,
            seed=args.seed,
            n_per_level=args.n_per_level,
            data_spec=project_cfg.data,
            control_spec=project_cfg.controls,
            concept_mode=args.concept_mode,
        )
    bundle = load_model_bundle(
        args.model,
        device=torch.device(args.device) if args.device else None,
        local_files_only=bool(args.local_files_only),
    )
    dataset = dataset_from_samples(samples)
    artifacts_dir, stats_dir, plots_dir = _get_dirs(args)
    cache_dir = Path(getattr(args, "cache_dir", artifacts_dir))
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )
    if getattr(args, "verbose", False):
        print("[geometry] computing deltas")
    residual = cache.residual
    deltas = compute_deltas(
        samples,
        residual,
        method="adjacent",
        concept_mode=args.concept_mode,
        topic_pair_strategy=args.topic_pair_strategy,
        pair_subsample_frac=args.pair_subsample_frac,
        seed=args.seed,
    )
    pca = compute_pca_metrics(deltas)
    rot = compute_rotation_metrics(pca.subspaces)

    if getattr(args, "verbose", False):
        print("[geometry] bootstrap CIs")
    rng = np.random.default_rng(args.seed)
    bands = bootstrap_curves(
        samples,
        residual,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
        verbose=bool(getattr(args, "verbose", False)),
    )

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
        title=f"PC1 variance ({concept}, {split}, {template_family})",
        ylabel="Explained variance",
        outpath=plots_dir / f"{stem}_pc1.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=pca.k_curves["k90"].astype(np.float32),
        low=bands["k90_low"],
        high=bands["k90_high"],
        label="k90",
        title=f"k90 ({concept}, {split}, {template_family})",
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
            title=f"Rotation ({concept}, {split}, {template_family})",
            ylabel="Rotation (deg)",
            outpath=plots_dir / f"{stem}_rotation.png",
        )
    _write_manifest(
        out_dir=stats_dir,
        name=f"{stem}_geometry",
        model_name=args.model,
        device=str(bundle.device),
        dataset_signature=dataset.dataset_signature,
        split_signature=hash_samples(samples),
        cache_key=cache.metadata.get("cache_key"),
        extra={"command": "geometry", "concept": concept, "split": split},
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
    artifacts_dir, stats_dir, plots_dir = _get_dirs(args)
    cache_dir = Path(getattr(args, "cache_dir", artifacts_dir))
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )
    residual = cache.residual

    base_deltas = compute_deltas(samples, residual, method="adjacent")
    base_pca = compute_pca_metrics(base_deltas)
    base_rot = compute_rotation_metrics(base_pca.subspaces)
    observed_pc1_mean = float(np.nanmean(base_pca.pc1_curve))
    observed_rot_mean = float(np.nanmean(base_rot.rotation_curve))

    if getattr(args, "verbose", False):
        print("[controls] building null distributions")
    rng = np.random.default_rng(args.seed)
    null_pc1 = np.zeros((args.n_shuffles,), dtype=np.float32)
    null_rot = np.zeros((args.n_shuffles,), dtype=np.float32)
    iterator = range(args.n_shuffles)
    if getattr(args, "verbose", False):
        iterator = tqdm(iterator, desc="permute controls", disable=False)
    for i in iterator:
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

    pc1_stats = empirical_p_stats(null_pc1, observed_pc1_mean)
    rot_stats = empirical_p_stats(null_rot, observed_rot_mean)

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
        {
            "pc1_mean": pc1_stats["p_two_tailed"],
            "rotation_mean": rot_stats["p_two_tailed"],
            "pc1_mean_p_two_tailed": pc1_stats["p_two_tailed"],
            "pc1_mean_p_hi": pc1_stats["p_hi"],
            "pc1_mean_p_lo": pc1_stats["p_lo"],
            "pc1_mean_effect_size": pc1_stats["effect_size"],
            "pc1_mean_z_like": pc1_stats["z_like"],
            "rotation_mean_p_two_tailed": rot_stats["p_two_tailed"],
            "rotation_mean_p_hi": rot_stats["p_hi"],
            "rotation_mean_p_lo": rot_stats["p_lo"],
            "rotation_mean_effect_size": rot_stats["effect_size"],
            "rotation_mean_z_like": rot_stats["z_like"],
        },
    )
    plot_null_hist(
        values=null_pc1,
        observed=observed_pc1_mean,
        title=(
            f"PC1 mean null ({concept}, {split}) "
            f"p={pc1_stats['p_two_tailed']:.3f} "
            f"eff={pc1_stats['effect_size']:.3f}"
        ),
        xlabel="PC1 mean",
        outpath=plots_dir / f"{stem}_null_pc1.png",
    )
    plot_null_hist(
        values=null_rot,
        observed=observed_rot_mean,
        title=(
            f"Rotation mean null ({concept}, {split}) "
            f"p={rot_stats['p_two_tailed']:.3f} "
            f"eff={rot_stats['effect_size']:.3f}"
        ),
        xlabel="Rotation mean (deg)",
        outpath=plots_dir / f"{stem}_null_rotation.png",
    )
    _write_manifest(
        out_dir=stats_dir,
        name=f"{stem}_controls",
        model_name=args.model,
        device=str(bundle.device),
        dataset_signature=dataset.dataset_signature,
        split_signature=hash_samples(samples),
        cache_key=cache.metadata.get("cache_key"),
        extra={"command": "controls", "concept": concept, "split": split},
    )


def cmd_topic_control(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.load_config(Path(args.config)) if args.config else cfgmod.ProjectConfig()
    concept = args.concept
    split = args.split
    template_family = _resolve_template_family(concept, args.template_family, project_cfg)
    control_family = args.control_template_family
    if concept != "sentiment":
        raise ValueError("topic_control is only implemented for sentiment concept")

    samples_sent = generate_samples(
        concept,
        split,
        template_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
        concept_mode="sentiment",
    )
    samples_ctrl = generate_samples(
        concept,
        split,
        control_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
        concept_mode="topic_control",
    )

    bundle = load_model_bundle(
        args.model,
        device=torch.device(args.device) if args.device else None,
        local_files_only=bool(args.local_files_only),
    )
    artifacts_dir, stats_dir, plots_dir = _get_dirs(args)
    cache_dir = Path(getattr(args, "cache_dir", artifacts_dir))

    dataset_sent = dataset_from_samples(samples_sent)
    dataset_ctrl = dataset_from_samples(samples_ctrl)
    cache_sent = build_or_load_activation_cache(
        bundle,
        dataset=dataset_sent,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )
    cache_ctrl = build_or_load_activation_cache(
        bundle,
        dataset=dataset_ctrl,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )

    if getattr(args, "verbose", False):
        print("[topic_control] computing deltas + directions")
    deltas_sent = compute_deltas(
        samples_sent,
        cache_sent.residual,
        method="adjacent",
        concept_mode="sentiment",
        seed=args.seed,
    )
    deltas_ctrl = compute_deltas(
        samples_ctrl,
        cache_ctrl.residual,
        method="adjacent",
        concept_mode="topic_control",
        topic_pair_strategy=args.topic_pair_strategy,
        pair_subsample_frac=args.pair_subsample_frac,
        seed=args.seed,
    )

    dirs_sent = pc1_directions_from_deltas(deltas_sent)
    dirs_ctrl = pc1_directions_from_deltas(deltas_ctrl)
    angles = angle_between_directions(dirs_sent, dirs_ctrl)
    explained_frac = explained_fraction_along_direction(deltas_ctrl, dirs_sent)

    safe_model = args.model.replace("/", "__")
    stem = f"{concept}__{split}__{control_family}__{safe_model}"
    np.savez_compressed(
        stats_dir / f"{stem}_topic_control.npz",
        angles_deg=angles,
        explained_frac_control=explained_frac,
    )
    _save_curves_csv(
        stats_dir / f"{stem}_topic_control.csv",
        curves={
            "angles_deg": angles,
            "explained_frac_control": explained_frac,
        },
    )
    save_json(
        stats_dir / f"{stem}_topic_control_summary.json",
        {
            "mean_angle_deg": float(np.nanmean(angles)),
            "mean_explained_frac_control": float(np.nanmean(explained_frac)),
            "control_template_family": control_family,
            "sentiment_template_family": template_family,
        },
    )

    layers = np.arange(len(angles))
    plot_curve_with_ci(
        x=layers,
        mean=angles,
        low=None,
        high=None,
        label="angle",
        title=f"Angle: sentiment vs topic-control ({concept}, {split})",
        ylabel="Angle (deg)",
        outpath=plots_dir / f"{stem}_topic_control_angles.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=explained_frac,
        low=None,
        high=None,
        label="explained",
        title=f"Control projection on sentiment PC1 ({concept}, {split})",
        ylabel="Explained fraction",
        outpath=plots_dir / f"{stem}_topic_control_explained.png",
    )

    _write_manifest(
        out_dir=stats_dir,
        name=f"{stem}_topic_control",
        model_name=args.model,
        device=str(bundle.device),
        dataset_signature=dataset_ctrl.dataset_signature,
        split_signature=hash_samples(samples_ctrl),
        cache_key=cache_ctrl.metadata.get("cache_key"),
        extra={
            "command": "topic_control",
            "concept": concept,
            "split": split,
            "control_template_family": control_family,
        },
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
    artifacts_dir, stats_dir, plots_dir = _get_dirs(args)
    if args.layer < 0:
        sweep = run_ablation_layer_sweep(
            samples,
            model_name=args.model,
            selection_method=args.method,
            m=m_list[0],
            alpha=args.alpha,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            artifacts_dir=Path(getattr(args, "cache_dir", artifacts_dir)),
            verbose=bool(getattr(args, "verbose", False)),
        )
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
        _write_manifest(
            out_dir=stats_dir,
            name=f"{stem}_ablate_layer_sweep",
            model_name=args.model,
            device=str(get_device()),
            dataset_signature=hash_samples(samples),
            split_signature=hash_samples(samples),
            cache_key=None,
            extra={"command": "ablate_layer_sweep", "concept": concept, "split": split},
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
        device=args.device,
        artifacts_dir=Path(getattr(args, "cache_dir", artifacts_dir)),
        verbose=bool(getattr(args, "verbose", False)),
    )
    safe_model = args.model.replace("/", "__")
    stem = f"{concept}__{split}__{template_family}__{safe_model}__L{args.layer}__{args.method}"
    save_ablation_artifacts(result, out_dir=stats_dir, stem=stem)
    plot_ablation_curves(result, out_dir=plots_dir, stem=stem)
    _write_manifest(
        out_dir=stats_dir,
        name=f"{stem}_ablate",
        model_name=args.model,
        device=str(torch.device(args.device)) if args.device else str(get_device()),
        dataset_signature=result.summary.get("dataset_signature"),
        split_signature=hash_samples(samples),
        cache_key=result.summary.get("cache_key"),
        extra={"command": "ablate", "concept": concept, "split": split},
    )


def cmd_specificity(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.ProjectConfig()
    concept_list = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if len(concept_list) != 2:
        raise ValueError("Expected exactly two concepts for specificity")
    concept_a, concept_b = concept_list
    template_family_a = _resolve_template_family(concept_a, args.template_family, project_cfg)
    template_family_b = _resolve_template_family(concept_b, args.template_family, project_cfg)
    samples_a = generate_samples(
        concept_a,
        args.split,
        template_family_a,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )
    samples_b = generate_samples(
        concept_b,
        args.split,
        template_family_b,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )

    bundle = load_model_bundle(
        args.model,
        device=torch.device(args.device) if args.device else None,
    )
    artifacts_dir, stats_dir, plots_dir = _get_dirs(args)
    dataset_a = dataset_from_samples(samples_a)
    dataset_b = dataset_from_samples(samples_b)
    cache_dir = Path(getattr(args, "cache_dir", artifacts_dir))
    cache_a = build_or_load_activation_cache(
        bundle,
        dataset=dataset_a,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )
    cache_b = build_or_load_activation_cache(
        bundle,
        dataset=dataset_b,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )

    if getattr(args, "verbose", False):
        print("[specificity] computing similarity + permutation")
    sim = similarity_across_concepts(samples_a, cache_a.residual, samples_b, cache_b.residual)
    perm = permutation_test_similarity(
        samples_a,
        cache_a.residual,
        samples_b,
        cache_b.residual,
        levels_a=project_cfg.data.concepts[concept_a].levels,
        levels_b=project_cfg.data.concepts[concept_b].levels,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        verbose=bool(getattr(args, "verbose", False)),
    )

    safe_model = args.model.replace("/", "__")
    stem = f"{concept_a}__{concept_b}__{args.split}__{safe_model}"
    np.savez_compressed(
        stats_dir / f"{stem}_similarity.npz",
        cosine=sim.cosine,
        angles_deg=sim.angles_deg,
        null_cosine=perm["null_cosine"],
        null_angles_deg=perm["null_angles_deg"],
        p_cosine=perm["p_cosine"],
        p_angles=perm["p_angles"],
        p_cosine_hi=perm["p_cosine_hi"],
        p_cosine_lo=perm["p_cosine_lo"],
        p_angles_hi=perm["p_angles_hi"],
        p_angles_lo=perm["p_angles_lo"],
        cosine_effect_size=perm["cosine_effect_size"],
        angles_effect_size=perm["angles_effect_size"],
        cosine_z_like=perm["cosine_z_like"],
        angles_z_like=perm["angles_z_like"],
    )

    layers = np.arange(len(sim.cosine))
    plot_curve_with_ci(
        x=layers,
        mean=sim.cosine,
        low=None,
        high=None,
        label="cosine",
        title=f"Direction cosine ({concept_a} vs {concept_b})",
        ylabel="Cosine similarity",
        outpath=plots_dir / f"{stem}_cosine.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=sim.angles_deg,
        low=None,
        high=None,
        label="angles",
        title=f"Subspace angles ({concept_a} vs {concept_b})",
        ylabel="Mean angle (deg)",
        outpath=plots_dir / f"{stem}_angles.png",
    )

    labels_a = np.array([int(s.metadata.get("level_id", -1)) for s in samples_a], dtype=np.float32)
    labels_b = np.array([int(s.metadata.get("level_id", -1)) for s in samples_b], dtype=np.float32)
    if (labels_b < 0).any():
        raise ValueError("Sample metadata missing level_id")
    if (labels_a < 0).any():
        raise ValueError("Sample metadata missing level_id")
    dirs_a = pc1_directions_from_samples(samples_a, cache_a.residual)
    dirs_b = pc1_directions_from_samples(samples_b, cache_b.residual)
    transfer_ab = transfer_direction(dirs_a, cache_b.residual, labels_b)
    transfer_ba = transfer_direction(dirs_b, cache_a.residual, labels_a)
    shared_ab = shared_variance_control(dirs_a, dirs_b, cache_b.residual, labels_b)
    transfer_ablation = cross_ablation_transfer(
        samples_a,
        samples_b,
        model_name=args.model,
        layer=args.ablate_layer,
        m=args.m,
        selection_method=args.method,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        artifacts_dir=Path("artifacts"),
    )
    np.savez_compressed(
        stats_dir / f"{stem}_transfer.npz",
        spearman_ab=transfer_ab.spearman,
        auc_extremes_ab=transfer_ab.auc_extremes,
        spearman_ba=transfer_ba.spearman,
        auc_extremes_ba=transfer_ba.auc_extremes,
        shared_control_ab=shared_ab,
        ablation_effect_a=np.array([transfer_ablation["effect_a"]], dtype=np.float32),
        ablation_effect_b=np.array([transfer_ablation["effect_b"]], dtype=np.float32),
    )
    transfer_matrix = np.array(
        [
            [float(np.nanmean(transfer_ab.spearman)), float(np.nanmean(transfer_ab.auc_extremes))],
            [float(np.nanmean(transfer_ba.spearman)), float(np.nanmean(transfer_ba.auc_extremes))],
        ],
        dtype=np.float32,
    )
    plot_heatmap(
        matrix=transfer_matrix,
        xlabels=["Spearman", "AUC"],
        ylabels=[f"{concept_a}->{concept_b}", f"{concept_b}->{concept_a}"],
        title="Transfer summary",
        outpath=plots_dir / f"{stem}_transfer_heatmap.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=transfer_ab.spearman,
        low=None,
        high=None,
        label="spearman",
        title=f"Transfer Spearman ({concept_a} -> {concept_b})",
        ylabel="Spearman",
        outpath=plots_dir / f"{stem}_transfer_spearman.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=transfer_ab.auc_extremes,
        low=None,
        high=None,
        label="auc",
        title=f"Transfer AUC ({concept_a} -> {concept_b})",
        ylabel="AUC",
        outpath=plots_dir / f"{stem}_transfer_auc.png",
    )
    plot_curve_with_ci(
        x=layers,
        mean=shared_ab,
        low=None,
        high=None,
        label="shared",
        title=f"Shared variance control ({concept_a}->{concept_b})",
        ylabel="Spearman (residualized)",
        outpath=plots_dir / f"{stem}_shared_variance.png",
    )
    plot_bar(
        labels=[f"{concept_a} effect", f"{concept_b} effect"],
        values=np.array(
            [transfer_ablation["effect_a"], transfer_ablation["effect_b"]], dtype=np.float32
        ),
        title="Cross ablation transfer",
        ylabel="Projection delta",
        outpath=plots_dir / f"{stem}_ablation_transfer.png",
    )
    _write_manifest(
        out_dir=stats_dir,
        name=f"{stem}_specificity",
        model_name=args.model,
        device=str(bundle.device),
        dataset_signature=None,
        split_signature=None,
        cache_key=None,
        extra={
            "command": "specificity",
            "concepts": [concept_a, concept_b],
            "split": args.split,
            "dataset_signature_a": dataset_a.dataset_signature,
            "dataset_signature_b": dataset_b.dataset_signature,
            "cache_key_a": cache_a.metadata.get("cache_key"),
            "cache_key_b": cache_b.metadata.get("cache_key"),
        },
    )


def cmd_behavior(args: argparse.Namespace) -> None:
    project_cfg = cfgmod.ProjectConfig()
    concept = args.concept
    template_family = _resolve_template_family(concept, args.template_family, project_cfg)
    samples_d = generate_samples(
        concept,
        "discovery",
        template_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )
    samples_e = generate_samples(
        concept,
        "eval",
        template_family,
        seed=args.seed,
        n_per_level=args.n_per_level,
        data_spec=project_cfg.data,
        control_spec=project_cfg.controls,
    )
    bundle = load_model_bundle(
        args.model,
        device=torch.device(args.device) if args.device else None,
    )
    artifacts_dir, stats_dir, plots_dir = _get_dirs(args)
    dataset_d = dataset_from_samples(samples_d)
    dataset_e = dataset_from_samples(samples_e)
    cache_dir = Path(getattr(args, "cache_dir", artifacts_dir))
    cache_d = build_or_load_activation_cache(
        bundle,
        dataset=dataset_d,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )
    cache_e = build_or_load_activation_cache(
        bundle,
        dataset=dataset_e,
        artifacts_dir=cache_dir,
        batch_size=args.batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=bool(args.use_cache),
        verbose=bool(getattr(args, "verbose", False)),
    )
    if getattr(args, "verbose", False):
        print("[behavior] training + eval probes")
    labels_d = np.array([int(s.metadata.get("level_id", -1)) for s in samples_d], dtype=np.float32)
    labels_e = np.array([int(s.metadata.get("level_id", -1)) for s in samples_e], dtype=np.float32)
    if (labels_d < 0).any() or (labels_e < 0).any():
        raise ValueError("Sample metadata missing level_id")

    probes = train_ridge_probes(cache_d.residual, labels_d, alpha=1.0)
    eval_spearman = eval_ridge_probes(
        cache_e.residual,
        labels_e,
        weights=probes.weights,
        intercepts=probes.intercepts,
    )

    safe_model = args.model.replace("/", "__")
    stem = f"{concept}__behavior__{safe_model}"
    np.savez_compressed(
        stats_dir / f"{stem}_probe.npz",
        train_spearman=probes.spearman_by_layer,
        eval_spearman=eval_spearman,
    )
    plot_curve_with_ci(
        x=np.arange(len(eval_spearman)),
        mean=eval_spearman,
        low=None,
        high=None,
        label="eval",
        title=f"Probe Spearman ({concept})",
        ylabel="Spearman",
        outpath=plots_dir / f"{stem}_probe_spearman.png",
    )

    impact = ablation_probe_impact(
        samples_e,
        model_name=args.model,
        layer=args.ablate_layer,
        m=args.m,
        selection_method=args.method,
        weights=probes.weights,
        intercepts=probes.intercepts,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        artifacts_dir=artifacts_dir,
    )
    save_json(stats_dir / f"{stem}_probe_ablation.json", impact)
    plot_bar(
        labels=["base", "ablated", "random"],
        values=np.array(
            [
                impact["base_spearman"],
                impact["ablated_spearman"],
                impact["random_spearman"],
            ],
            dtype=np.float32,
        ),
        title="Probe Spearman (base vs ablated vs random)",
        ylabel="Spearman",
        outpath=plots_dir / f"{stem}_probe_ablation.png",
    )
    _write_manifest(
        out_dir=stats_dir,
        name=f"{stem}_behavior",
        model_name=args.model,
        device=str(bundle.device),
        dataset_signature=None,
        split_signature=None,
        cache_key=None,
        extra={
            "command": "behavior",
            "concept": concept,
            "dataset_signature_discovery": dataset_d.dataset_signature,
            "dataset_signature_eval": dataset_e.dataset_signature,
            "cache_key_discovery": cache_d.metadata.get("cache_key"),
            "cache_key_eval": cache_e.metadata.get("cache_key"),
        },
    )


def cmd_run_all(args: argparse.Namespace) -> None:
    if args.backend != "nnsight":
        raise ValueError("Only nnsight backend is supported")
    cfg = load_config(Path(args.config) if args.config else None)
    raw = {}
    if args.config:
        with Path(args.config).open() as f:
            raw = yaml.safe_load(f) or {}

    run_dir = ensure_run_dir(cfg.artifacts_dir, cfg.run_id)
    save_config_snapshot(run_dir, cfg)
    write_run_metadata(run_dir, cfg, seeds=cfg.seeds)

    model_names = raw.get("model_names", [cfg.model_name])
    concepts = raw.get("concepts", ["sentiment", "concreteness"])
    if args.use_cache is not None:
        use_cache = bool(args.use_cache)
    else:
        use_cache = bool(raw.get("use_cache", True))
    batch_size = int(raw.get("batch_size", cfg.batch_size))
    n_per_level = int(raw.get("n_per_level", 2))
    seed = int(raw.get("seed", 0))
    geom_bootstrap = int(raw.get("geometry", {}).get("n_bootstrap", 200))
    spec_shuffles = int(raw.get("specificity", {}).get("n_shuffles", 50))
    ablation_cfg = raw.get("ablation", {})
    behavior_cfg = raw.get("behavior", {})

    stats_dir = run_dir / "stats"
    plots_dir = run_dir / "plots"
    ensure_dir(stats_dir)
    ensure_dir(plots_dir)

    full_all_models = bool(raw.get("run_all_full_all_models", False))
    device = args.device
    use_verbose = bool(getattr(args, "verbose", False))
    model_iter = model_names
    if use_verbose:
        model_iter = tqdm(model_names, desc="models", disable=False)

    for idx, model_name in enumerate(model_iter):
        full = full_all_models or idx == 0
        print(f"[run_all] model={model_name} full={full}")
        concept_iter = concepts
        if use_verbose:
            concept_iter = tqdm(concepts, desc=f"concepts:{model_name}", leave=False, disable=False)
        for concept in concept_iter:
            print(f"[run_all] geometry concept={concept} split=discovery")
            geom_args = argparse.Namespace(
                config=None,
                concept=concept,
                split="discovery",
                template_family=None,
                n_per_level=n_per_level,
                seed=seed,
                model=model_name,
                batch_size=batch_size,
                device=device,
                use_cache=int(use_cache),
                n_bootstrap=geom_bootstrap,
                local_files_only=1,
                artifacts_dir=run_dir,
                cache_dir=cfg.artifacts_dir,
                concept_mode="sentiment",
                topic_pair_strategy="cartesian",
                pair_subsample_frac=None,
                verbose=int(use_verbose),
            )
            cmd_geometry(geom_args)

            print(f"[run_all] geometry concept={concept} split=discovery templates=aggregated")
            agg_geom_args = argparse.Namespace(
                config=None,
                concept=concept,
                split="discovery",
                template_family="aggregated-templates",
                n_per_level=n_per_level,
                seed=seed,
                model=model_name,
                batch_size=batch_size,
                device=device,
                use_cache=int(use_cache),
                n_bootstrap=geom_bootstrap,
                local_files_only=1,
                artifacts_dir=run_dir,
                cache_dir=cfg.artifacts_dir,
                concept_mode="sentiment",
                topic_pair_strategy="cartesian",
                pair_subsample_frac=None,
                verbose=int(use_verbose),
            )
            cmd_geometry(agg_geom_args)

            print(f"[run_all] controls concept={concept} split=discovery")
            ctrl_args = argparse.Namespace(
                config=None,
                concept=concept,
                split="discovery",
                template_family=None,
                n_per_level=n_per_level,
                seed=seed,
                model=model_name,
                batch_size=batch_size,
                device=device,
                use_cache=int(use_cache),
                n_shuffles=spec_shuffles,
                local_files_only=1,
                artifacts_dir=run_dir,
                cache_dir=cfg.artifacts_dir,
                verbose=int(use_verbose),
            )
            cmd_controls(ctrl_args)

            if full and concept == "sentiment":
                print("[run_all] geometry topic_control split=discovery")
                topic_geom_args = argparse.Namespace(
                    config=None,
                    concept=concept,
                    split="discovery",
                    template_family="topic_swap_fixed_sentiment",
                    n_per_level=n_per_level,
                    seed=seed,
                    model=model_name,
                    batch_size=batch_size,
                    device=device,
                    use_cache=int(use_cache),
                    n_bootstrap=geom_bootstrap,
                    local_files_only=1,
                    artifacts_dir=run_dir,
                    cache_dir=cfg.artifacts_dir,
                    concept_mode="topic_control",
                    topic_pair_strategy="cartesian",
                    pair_subsample_frac=None,
                    verbose=int(use_verbose),
                )
                cmd_geometry(topic_geom_args)

                print("[run_all] topic_control comparison split=discovery")
                topic_ctrl_args = argparse.Namespace(
                    config=None,
                    concept=concept,
                    split="discovery",
                    template_family="adjective_clause",
                    control_template_family="topic_swap_fixed_sentiment",
                    n_per_level=n_per_level,
                    seed=seed,
                    model=model_name,
                    batch_size=batch_size,
                    device=device,
                    use_cache=int(use_cache),
                    local_files_only=1,
                    artifacts_dir=run_dir,
                    cache_dir=cfg.artifacts_dir,
                    topic_pair_strategy="cartesian",
                    pair_subsample_frac=None,
                    verbose=int(use_verbose),
                )
                cmd_topic_control(topic_ctrl_args)

            if full:
                print(f"[run_all] ablate concept={concept} split=eval")
                ablate_args = argparse.Namespace(
                    concept=concept,
                    split="eval",
                    template_family=None,
                    n_per_level=n_per_level,
                    seed=seed,
                    model=model_name,
                    batch_size=batch_size,
                    layer=int(ablation_cfg.get("layer", 2)),
                    method=ablation_cfg.get("method", "variance"),
                    m_list=",".join(str(x) for x in ablation_cfg.get("m_list", [5, 10, 20])),
                    alpha=0.05,
                    random_control=1,
                    device=device,
                    artifacts_dir=run_dir,
                    cache_dir=cfg.artifacts_dir,
                    verbose=int(use_verbose),
                )
                cmd_ablate(ablate_args)

                print(f"[run_all] behavior concept={concept}")
                behavior_args = argparse.Namespace(
                    concept=concept,
                    template_family=None,
                    n_per_level=n_per_level,
                    seed=seed,
                    model=model_name,
                    batch_size=batch_size,
                    use_cache=int(use_cache),
                    ablate_layer=int(behavior_cfg.get("ablate_layer", 2)),
                    m=int(behavior_cfg.get("m", 20)),
                    method=behavior_cfg.get("method", "probe_weight"),
                    device=device,
                    artifacts_dir=run_dir,
                    cache_dir=cfg.artifacts_dir,
                    verbose=int(use_verbose),
                )
                cmd_behavior(behavior_args)

        if "sentiment" in concepts and "concreteness" in concepts:
            if full:
                print("[run_all] specificity sentiment,concreteness split=discovery")
                spec_args = argparse.Namespace(
                    concepts="sentiment,concreteness",
                    split="discovery",
                    template_family=None,
                    n_per_level=n_per_level,
                    seed=seed,
                    model=model_name,
                    batch_size=batch_size,
                    use_cache=int(use_cache),
                    n_shuffles=spec_shuffles,
                    ablate_layer=int(ablation_cfg.get("layer", 2)),
                    m=int(ablation_cfg.get("m_list", [20])[0] if ablation_cfg.get("m_list") else 20),
                    method=ablation_cfg.get("method", "variance"),
                    device=device,
                    artifacts_dir=run_dir,
                    cache_dir=cfg.artifacts_dir,
                    verbose=int(use_verbose),
                )
                cmd_specificity(spec_args)

    render_report(
        run_dir,
        {
            "run_dir": str(run_dir),
            "models": model_names,
            "concepts": concepts,
        },
    )
    write_methods(run_dir)
    write_reproducibility(run_dir, cfg)
    _export_paper_figures(run_dir)
    _write_manifest(
        out_dir=run_dir / "stats",
        name="run_all",
        model_name="multiple",
        device=str(torch.device(device)) if device else str(get_device()),
        dataset_signature=None,
        split_signature=None,
        cache_key=None,
        extra={"command": "run_all", "models": model_names, "concepts": concepts},
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Concept-paths research runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_all = sub.add_parser("run_all", help="Run full pipeline (geometry + controls + ablation)")
    run_all.add_argument("--config", type=str, default=None, help="Path to YAML config")
    run_all.add_argument("--backend", type=str, default="nnsight")
    run_all.add_argument("--use_cache", type=int, default=None)
    run_all.add_argument("--device", type=str, default=None)
    run_all.add_argument("--verbose", type=int, default=0)
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
    geometry.add_argument("--verbose", type=int, default=0)
    geometry.add_argument(
        "--concept_mode",
        type=str,
        default="sentiment",
        choices=["sentiment", "unordered", "topic_control"],
    )
    geometry.add_argument(
        "--topic_pair_strategy",
        type=str,
        default="cartesian",
        choices=["cartesian", "random"],
    )
    geometry.add_argument("--pair_subsample_frac", type=float, default=None)
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
    controls.add_argument("--verbose", type=int, default=0)
    controls.set_defaults(func=cmd_controls)

    topic_control = sub.add_parser(
        "topic_control",
        help="Concept-neutral topic-swap control (fixed adjective, varied topic)",
    )
    topic_control.add_argument("--config", type=str, default=None, help="Path to YAML config")
    topic_control.add_argument("--concept", type=str, default="sentiment")
    topic_control.add_argument("--split", type=str, default="discovery")
    topic_control.add_argument("--template_family", type=str, default=None)
    topic_control.add_argument("--control_template_family", type=str, default="topic_swap_fixed_sentiment")
    topic_control.add_argument("--n_per_level", type=int, default=2)
    topic_control.add_argument("--seed", type=int, default=0)
    topic_control.add_argument("--model", type=str, default="distilgpt2")
    topic_control.add_argument("--batch_size", type=int, default=16)
    topic_control.add_argument("--device", type=str, default=None)
    topic_control.add_argument("--use_cache", type=int, default=1)
    topic_control.add_argument("--local_files_only", type=int, default=1)
    topic_control.add_argument(
        "--topic_pair_strategy",
        type=str,
        default="cartesian",
        choices=["cartesian", "random"],
    )
    topic_control.add_argument("--pair_subsample_frac", type=float, default=None)
    topic_control.add_argument("--verbose", type=int, default=0)
    topic_control.set_defaults(func=cmd_topic_control)

    ablate = sub.add_parser("ablate", help="Run ablation pipeline")
    ablate.add_argument("--concept", type=str, default="sentiment")
    ablate.add_argument("--split", type=str, default="eval")
    ablate.add_argument("--template_family", type=str, default=None)
    ablate.add_argument("--n_per_level", type=int, default=2)
    ablate.add_argument("--seed", type=int, default=0)
    ablate.add_argument("--model", type=str, default="distilgpt2")
    ablate.add_argument("--batch_size", type=int, default=8)
    ablate.add_argument("--device", type=str, default=None)
    ablate.add_argument("--layer", type=int, required=True)
    ablate.add_argument("--method", type=str, default="variance")
    ablate.add_argument("--m_list", type=str, default="5,10,20,40,80")
    ablate.add_argument("--alpha", type=float, default=0.05)
    ablate.add_argument("--random_control", type=int, default=1)
    ablate.add_argument("--verbose", type=int, default=0)
    ablate.set_defaults(func=cmd_ablate)

    specificity = sub.add_parser("specificity", help="Direction specificity + transfer tests")
    specificity.add_argument("--concepts", type=str, required=True)
    specificity.add_argument("--split", type=str, default="discovery")
    specificity.add_argument("--template_family", type=str, default=None)
    specificity.add_argument("--n_per_level", type=int, default=2)
    specificity.add_argument("--seed", type=int, default=0)
    specificity.add_argument("--model", type=str, default="distilgpt2")
    specificity.add_argument("--batch_size", type=int, default=8)
    specificity.add_argument("--device", type=str, default=None)
    specificity.add_argument("--use_cache", type=int, default=1)
    specificity.add_argument("--n_shuffles", type=int, default=50)
    specificity.add_argument("--ablate_layer", type=int, default=2)
    specificity.add_argument("--m", type=int, default=20)
    specificity.add_argument("--method", type=str, default="variance")
    specificity.add_argument("--verbose", type=int, default=0)
    specificity.set_defaults(func=cmd_specificity)

    behavior = sub.add_parser("behavior", help="Ridge probe behavior + ablation impact")
    behavior.add_argument("--concept", type=str, default="sentiment")
    behavior.add_argument("--template_family", type=str, default=None)
    behavior.add_argument("--n_per_level", type=int, default=2)
    behavior.add_argument("--seed", type=int, default=0)
    behavior.add_argument("--model", type=str, default="distilgpt2")
    behavior.add_argument("--batch_size", type=int, default=8)
    behavior.add_argument("--device", type=str, default=None)
    behavior.add_argument("--use_cache", type=int, default=1)
    behavior.add_argument("--ablate_layer", type=int, default=2)
    behavior.add_argument("--m", type=int, default=20)
    behavior.add_argument("--method", type=str, default="probe_weight")
    behavior.add_argument("--verbose", type=int, default=0)
    behavior.set_defaults(func=cmd_behavior)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
