from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .. import config as cfgmod
from ..activations import ActivationCache, build_or_load_activation_cache, load_lm
from ..geometry import (
    PCAMetrics,
    deltas_from_pairs,
    pca_metrics_by_layer,
    random_direction_baseline,
    random_subspace_rotation,
    rotation_by_layer,
)
from ..io import ensure_run_dir, save_config_snapshot, save_npz, write_manifest, write_run_metadata
from ..plots import plot_metric_by_layer, plot_with_band
from ..stimuli import DeltaPair, generate_stimuli, prompt_hash_manifest, subsample_pairs, build_delta_pairs
from ..utils import ensure_dir, set_seed
from ..ablate import (
    build_or_load_mlp_act_cache,
    concept_direction_pc1,
    select_neurons,
    eval_ablation_effect,
    save_neuron_selections_csv,
)


def _ordinal_labels(pairs: List[DeltaPair]) -> np.ndarray:
    return np.array([p.pos_level_id for p in pairs], dtype=np.float32)


def run_geometry_once(
    *,
    cfg: cfgmod.RunConfig,
    seed: int,
    permute_labels: bool = False,
    concept_mode: str = "sentiment",
    pair_subsample_frac: Optional[float] = None,
    run_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    set_seed(seed)
    lm = load_lm(cfg.model_name, local_files_only=cfg.local_files_only, device_override=cfg.device)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    d_prompts: Optional[list[str]] = None
    e_prompts: Optional[list[str]] = None

    for split in ["discovery", "eval"]:
        samples = generate_stimuli(
            cfg.stimuli,
            split=split,  # type: ignore[arg-type]
            permute_labels=permute_labels,
            concept_mode=concept_mode,  # type: ignore[arg-type]
            seed=seed,
        )
        prompts = [s.prompt for s in samples]
        keys = [s.key for s in samples]
        pairs = build_delta_pairs(samples, strategy=cfg.geometry.delta_pair_strategy, seed=seed)
        pairs = subsample_pairs(pairs, pair_subsample_frac, seed)  # type: ignore[arg-type]

        # Leakage guard: discovery vs eval prompt hashes disjoint
        if split == "discovery":
            d_prompts = prompts
        if split == "eval":
            e_prompts = prompts

        cache = build_or_load_activation_cache(
            lm,
            keys=keys,
            prompts=prompts,
            artifacts_dir=cfg.artifacts_dir,
            split=split,
            model_name=cfg.model_name,
            seed=seed,
            batch_size=cfg.batch_size,
            device=None,
            force_recompute=False,
        )
        deltas = deltas_from_pairs(keys=cache.keys, acts=cache.acts, pairs=pairs)
        ordinal = _ordinal_labels(pairs) if concept_mode == "sentiment" else None
        pmetrics = pca_metrics_by_layer(
            deltas, solver=cfg.geometry.pca_solver, ordinal=ordinal, anchor=cfg.geometry.pc1_anchor
        )
        rot = rotation_by_layer(
            deltas,
            solver=cfg.geometry.pca_solver,
            k_mode=cfg.geometry.rotation_k_mode,
            k_fixed=cfg.geometry.rotation_k_fixed,
            k90=pmetrics.k90,
            metric=cfg.geometry.rotation_metric,
        )
        rng = np.random.default_rng(seed)
        rand_proj = random_direction_baseline(
            deltas, n=cfg.geometry.random_baseline_directions, rng=rng
        )
        rand_rot = random_subspace_rotation(
            deltas, n=cfg.geometry.random_baseline_subspaces, k=pmetrics.k90, rng=rng
        )

        results[split] = {
            "top_pc": pmetrics.top_pc_ratio,
            "k90": pmetrics.k90.astype(np.float32),
            "rotation": rot,
            "rand_proj_mean": rand_proj.mean(axis=0),
            "rand_proj_std": rand_proj.std(axis=0),
            "rand_rot_mean": rand_rot.mean(axis=0),
            "rand_rot_std": rand_rot.std(axis=0),
        }

        # Save per-run data
        split_dir = run_dir / f"seed{seed}" / split
        ensure_dir(split_dir)
        save_npz(split_dir / "pca.npz", top_pc=pmetrics.top_pc_ratio, k90=pmetrics.k90)
        save_npz(split_dir / "rotation.npz", rotation=rot)
        save_npz(split_dir / "random_baseline.npz", rand_proj_mean=rand_proj.mean(axis=0), rand_proj_std=rand_proj.std(axis=0))
        manifest = {
            "permute_labels": permute_labels,
            "concept_mode": concept_mode,
            "n_pairs": int(deltas.shape[0]),
            "n_layers": int(deltas.shape[1]),
            "hidden": int(deltas.shape[2]),
        }
        write_manifest(split_dir / "manifest.json", manifest)

    # Cross-split leakage guard
    if d_prompts is not None and e_prompts is not None:
        prompt_hashes = prompt_hash_manifest(d_prompts, e_prompts)
        write_manifest(run_dir / f"seed{seed}" / "prompt_hashes.json", prompt_hashes)
    return results


def aggregate_curves(curves: List[np.ndarray]) -> Dict[str, np.ndarray]:
    stacked = np.stack(curves, axis=0)
    return {"mean": stacked.mean(axis=0), "std": stacked.std(axis=0)}


def run_geometry(
    cfg: cfgmod.RunConfig,
    *,
    permute_labels: bool = False,
    concept_mode: str = "sentiment",
    run_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    per_seed_results = []
    for seed in cfg.seeds:
        res = run_geometry_once(
            cfg=cfg,
            seed=seed,
            permute_labels=permute_labels,
            concept_mode=concept_mode,
            pair_subsample_frac=cfg.geometry.pair_subsample_frac,
            run_dir=run_dir,
        )
        per_seed_results.append(res)

    # Aggregate
    agg: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ["discovery", "eval"]:
        top_pc_curves = [r[split]["top_pc"] for r in per_seed_results]
        k90_curves = [r[split]["k90"] for r in per_seed_results]
        rot_curves = [r[split]["rotation"] for r in per_seed_results]
        agg[split] = {
            "top_pc_mean": aggregate_curves(top_pc_curves)["mean"],
            "top_pc_std": aggregate_curves(top_pc_curves)["std"],
            "k90_mean": aggregate_curves(k90_curves)["mean"],
            "k90_std": aggregate_curves(k90_curves)["std"],
            "rotation_mean": aggregate_curves(rot_curves)["mean"],
            "rotation_std": aggregate_curves(rot_curves)["std"],
        }
    return agg


def run_ablation(
    cfg: cfgmod.RunConfig,
    *,
    run_dir: Path,
    selector: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    set_seed(seed)
    lm = load_lm(cfg.model_name, local_files_only=cfg.local_files_only, device_override=cfg.device)

    # Discovery for selection
    d_samples = generate_stimuli(cfg.stimuli, split="discovery", permute_labels=False, concept_mode="sentiment", seed=seed)  # type: ignore[arg-type]
    d_prompts = [s.prompt for s in d_samples]
    d_keys = [s.key for s in d_samples]
    d_pairs = build_delta_pairs(d_samples, strategy=cfg.geometry.delta_pair_strategy, seed=seed)
    d_cache = build_or_load_activation_cache(
        lm,
        keys=d_keys,
        prompts=d_prompts,
        artifacts_dir=cfg.artifacts_dir,
        split="discovery",
        model_name=cfg.model_name,
        seed=seed,
        batch_size=cfg.batch_size,
        device=None,
        force_recompute=False,
    )
    d_deltas = deltas_from_pairs(keys=d_cache.keys, acts=d_cache.acts, pairs=d_pairs)

    # Eval for measuring effect
    e_samples = generate_stimuli(cfg.stimuli, split="eval", permute_labels=False, concept_mode="sentiment", seed=seed)  # type: ignore[arg-type]
    e_prompts = [s.prompt for s in e_samples]
    e_keys = [s.key for s in e_samples]
    e_pairs = build_delta_pairs(e_samples, strategy=cfg.geometry.delta_pair_strategy, seed=seed)
    e_cache = build_or_load_activation_cache(
        lm,
        keys=e_keys,
        prompts=e_prompts,
        artifacts_dir=cfg.artifacts_dir,
        split="eval",
        model_name=cfg.model_name,
        seed=seed,
        batch_size=cfg.batch_size,
        device=None,
        force_recompute=False,
    )

    n_layers = d_cache.acts.shape[1]
    effects = np.zeros((n_layers - 1,), dtype=np.float32)
    rand_effects = np.zeros((n_layers - 1,), dtype=np.float32)
    anti_effects = np.zeros((n_layers - 1,), dtype=np.float32)
    selections = []
    ordinal = _ordinal_labels(d_pairs)

    for layer in range(n_layers - 1):
        # Selection
        d_mlp_act = build_or_load_mlp_act_cache(
            lm,
            keys=d_keys,
            prompts=d_prompts,
            artifacts_dir=cfg.artifacts_dir,
            split="discovery",
            model_name=cfg.model_name,
            seed=seed,
            layer=layer,
            batch_size=cfg.batch_size,
            device=None,
            force_recompute=False,
        )
        d_mlp_deltas = deltas_from_pairs(keys=d_keys, acts=d_mlp_act[:, None, :], pairs=d_pairs)[:, 0, :]
        sel = select_neurons(
            selector=selector,
            deltas_resid=d_deltas,
            deltas_resid_next=d_deltas[:, layer + 1, :],
            deltas_mlp=d_mlp_deltas,
            layer=layer,
            top_m=cfg.ablation.top_m,
            ordinal=ordinal,
        )
        selections.append(sel)

        concept_dir = concept_direction_pc1(d_deltas[:, layer, :])
        scores_abs = np.abs(sel.score)
        order_all = np.argsort(scores_abs)
        anti_idx = order_all[: len(sel.neuron_idx)]

        res = eval_ablation_effect(
            keys=e_keys,
            acts=e_cache.acts,
            pairs=e_pairs,
            lm=lm,
            prompts=e_prompts,
            selection=sel,
            concept_dir=concept_dir,
            d_mlp=int(d_mlp_act.shape[1]),
            batch_size=cfg.batch_size,
            device=None,
            rng=np.random.default_rng(seed),
            anti_indices=anti_idx,
        )
        effects[layer] = res["effect"]
        rand_effects[layer] = res["random_effect"]
        anti_effects[layer] = res.get("anti_effect", np.nan)

    save_neuron_selections_csv(
        selections, run_dir / f"ablation_selections_{selector}_seed{seed}.csv"
    )
    save_npz(
        run_dir / f"ablation_effects_{selector}_seed{seed}.npz",
        effect=effects,
        random_effect=rand_effects,
        anti_effect=anti_effects,
    )
    return {
        "effect": effects,
        "random_effect": rand_effects,
        "anti_effect": anti_effects,
    }
