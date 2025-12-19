from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .capture import capture_activations, _select_adapter
from .config import ControlSpec, DataSpec, ExperimentConfig
from .data import Sample, generate_samples
from .plots import plot_curves_with_ci, plot_bar
from .selection import select_neurons
from .utils import ensure_dir, save_json


def fit_layerwise_ridge(
    residual: np.ndarray, labels: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    n_layers = residual.shape[1]
    scores = np.zeros((n_layers,), dtype=np.float32)
    for l in range(n_layers):
        x = residual[:, l, :]
        model = Ridge(alpha=alpha)
        model.fit(x, labels)
        pred = model.predict(x)
        r, _ = spearmanr(pred, labels)
        scores[l] = r if np.isfinite(r) else 0.0
    return scores


def run_behavior(
    cfg: ExperimentConfig,
    *,
    concept: str,
    adapter: str,
    model: str,
    artifacts_dir: Path,
    use_cache: bool,
) -> None:
    ensure_dir(artifacts_dir / "plots")
    ensure_dir(artifacts_dir / "stats")
    ds_disc = DataSpec(concept=concept, split="discovery", template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
    ds_eval = DataSpec(concept=concept, split="eval", template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
    ctrl = ControlSpec()
    samples_disc, _ = generate_samples(cfg, data_spec=ds_disc, control=ctrl)
    samples_eval, _ = generate_samples(cfg, data_spec=ds_eval, control=ctrl)
    cache_disc = capture_activations(
        config=cfg,
        data_spec=ds_disc,
        control_spec=ctrl,
        adapter_name=adapter,
        model_name=model,
        batch_size=cfg.data.n_per_level,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    cache_eval = capture_activations(
        config=cfg,
        data_spec=ds_eval,
        control_spec=ctrl,
        adapter_name=adapter,
        model_name=model,
        batch_size=cfg.data.n_per_level,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    labels_disc = np.array([s.level for s in samples_disc], dtype=np.float32)
    labels_eval = np.array([s.level for s in samples_eval], dtype=np.float32)
    scores_disc = fit_layerwise_ridge(cache_disc.residual, labels_disc, alpha=1.0)
    scores_eval = fit_layerwise_ridge(cache_eval.residual, labels_eval, alpha=1.0)
    x = np.arange(scores_disc.shape[0])
    plot_curves_with_ci(
        x=x,
        mean=scores_eval,
        ci_low=scores_eval,
        ci_high=scores_eval,
        label="eval ridge",
        title=f"Probe performance ({concept})",
        ylabel="Spearman",
        outpath=artifacts_dir / "plots" / f"probe_{concept}.png",
        raw_out=artifacts_dir / "raw" / f"probe_{concept}.npz",
    )
    save_json(
        artifacts_dir / "stats" / f"probe_{concept}.json",
        {"scores_eval": scores_eval.tolist(), "scores_disc": scores_disc.tolist()},
    )


def ablation_impact_on_behavior(
    cfg: ExperimentConfig,
    *,
    concept: str,
    adapter_name: str,
    model: str,
    layer: int,
    m: int,
    selection_method: str,
    artifacts_dir: Path,
    use_cache: bool,
) -> None:
    ctrl = ControlSpec()
    ds_disc = DataSpec(concept=concept, split="discovery", template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
    ds_eval = DataSpec(concept=concept, split="eval", template_family=cfg.data.template_family, seed=cfg.data.seed, n_levels=cfg.data.n_levels, n_per_level=cfg.data.n_per_level)
    samples_disc, _ = generate_samples(cfg, data_spec=ds_disc, control=ctrl)
    samples_eval, _ = generate_samples(cfg, data_spec=ds_eval, control=ctrl)
    cache_disc = capture_activations(
        config=cfg,
        data_spec=ds_disc,
        control_spec=ctrl,
        adapter_name=adapter_name,
        model_name=model,
        batch_size=cfg.data.n_per_level,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    cache_eval = capture_activations(
        config=cfg,
        data_spec=ds_eval,
        control_spec=ctrl,
        adapter_name=adapter_name,
        model_name=model,
        batch_size=cfg.data.n_per_level,
        artifacts_dir=artifacts_dir,
        use_cache=use_cache,
        local_files_only=True,
    )
    labels_disc = np.array([s.level for s in samples_disc], dtype=np.float32)
    idx = select_neurons(
        method=selection_method,
        mlp_acts=cache_disc.mlp,
        labels=labels_disc,
        layer=layer,
        m=m,
        seed=cfg.data.seed,
    )
    adapter = _select_adapter(adapter_name, model, local_files_only=True)
    prompts_eval = [s.prompt_text for s in samples_eval]
    base_resid = cache_eval.residual
    abl_resid = adapter.capture(prompts_eval, batch_size=cfg.data.n_per_level, ablate_layer=layer, neuron_idx=idx).residual
    rand_idx = np.random.default_rng(cfg.data.seed).choice(cache_disc.mlp.shape[2], size=len(idx), replace=False)
    rand_resid = adapter.capture(prompts_eval, batch_size=cfg.data.n_per_level, ablate_layer=layer, neuron_idx=rand_idx).residual
    labels_eval = np.array([s.level for s in samples_eval], dtype=np.float32)
    base_scores = fit_layerwise_ridge(base_resid, labels_eval, alpha=1.0)
    abl_scores = fit_layerwise_ridge(abl_resid, labels_eval, alpha=1.0)
    rand_scores = fit_layerwise_ridge(rand_resid, labels_eval, alpha=1.0)
    effect_target = base_scores - abl_scores
    effect_rand = base_scores - rand_scores
    plot_bar(
        labels=[f"L{layer} target", f"L{layer} random"],
        values=[effect_target[layer], effect_rand[layer]],
        title="Behavior drop from ablation",
        ylabel="Δ Spearman",
        outpath=artifacts_dir / "plots" / f"behavior_ablation_{concept}.png",
        raw_out=artifacts_dir / "raw" / f"behavior_ablation_{concept}.npz",
    )
    save_json(
        artifacts_dir / "stats" / f"behavior_ablation_{concept}.json",
        {
            "effect_target": effect_target.tolist(),
            "effect_random": effect_rand.tolist(),
            "idx": idx.tolist(),
        },
    )
