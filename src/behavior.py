from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from .ablation import capture_residual_with_ablation
from .capture import build_or_load_activation_cache, dataset_from_samples, load_model_bundle
from .selection import select_neurons


@dataclass(frozen=True)
class ProbeResult:
    spearman_by_layer: np.ndarray
    weights: list[np.ndarray]
    intercepts: list[float]


def _ensure_decoder_only(model: Any) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model config missing; cannot verify decoder-only")
    if getattr(cfg, "is_encoder_decoder", False):
        raise ValueError("Behavior probes require decoder-only models")


def train_ridge_probes(
    residual: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 1.0,
) -> ProbeResult:
    n_layers = residual.shape[0]
    weights = []
    intercepts = []
    spears = np.zeros((n_layers,), dtype=np.float32)

    for layer in range(n_layers):
        x = residual[layer]
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(x, labels)
        pred = model.predict(x)
        corr = spearmanr(pred, labels).correlation
        spears[layer] = float(corr) if corr is not None else np.nan
        weights.append(model.coef_.astype(np.float32))
        intercepts.append(float(model.intercept_))

    return ProbeResult(spearman_by_layer=spears, weights=weights, intercepts=intercepts)


def eval_ridge_probes(
    residual: np.ndarray,
    labels: np.ndarray,
    *,
    weights: Sequence[np.ndarray],
    intercepts: Sequence[float],
) -> np.ndarray:
    n_layers = residual.shape[0]
    out = np.zeros((n_layers,), dtype=np.float32)
    for layer in range(n_layers):
        pred = residual[layer] @ weights[layer] + intercepts[layer]
        corr = spearmanr(pred, labels).correlation
        out[layer] = float(corr) if corr is not None else np.nan
    return out


def ablation_probe_impact(
    samples_eval,
    *,
    model_name: str,
    layer: int,
    m: int,
    selection_method: str,
    weights: Sequence[np.ndarray],
    intercepts: Sequence[float],
    batch_size: int,
    seed: int,
    device: str | None,
    local_files_only: bool = True,
    artifacts_dir,
) -> Dict[str, float]:
    bundle = load_model_bundle(
        model_name,
        device=torch.device(device) if device else None,
        local_files_only=local_files_only,
    )
    _ensure_decoder_only(bundle.model)
    dataset = dataset_from_samples(samples_eval)
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=artifacts_dir,
        batch_size=batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=True,
    )
    labels = np.array([int(s.metadata.get("level_id", -1)) for s in samples_eval], dtype=np.float32)
    if (labels < 0).any():
        raise ValueError("Sample metadata missing level_id")

    mlp_layer = cache.mlp[layer]
    idx = select_neurons(selection_method, mlp_layer, labels, m=m, seed=seed)
    capture_layer = min(layer + 1, cache.residual.shape[0] - 1)

    base_pred = cache.residual[capture_layer] @ weights[capture_layer] + intercepts[capture_layer]
    base_corr = spearmanr(base_pred, labels).correlation

    ablated_resid = capture_residual_with_ablation(
        bundle,
        dataset.prompts,
        ablate_layer=layer,
        neuron_idx=idx,
        capture_layer=capture_layer,
        batch_size=batch_size,
    )
    ablated_pred = ablated_resid @ weights[capture_layer] + intercepts[capture_layer]
    ablated_corr = spearmanr(ablated_pred, labels).correlation

    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(mlp_layer.shape[1], size=m, replace=False)
    rand_resid = capture_residual_with_ablation(
        bundle,
        dataset.prompts,
        ablate_layer=layer,
        neuron_idx=rand_idx,
        capture_layer=capture_layer,
        batch_size=batch_size,
    )
    rand_pred = rand_resid @ weights[capture_layer] + intercepts[capture_layer]
    rand_corr = spearmanr(rand_pred, labels).correlation

    return {
        "base_spearman": float(base_corr) if base_corr is not None else np.nan,
        "ablated_spearman": float(ablated_corr) if ablated_corr is not None else np.nan,
        "random_spearman": float(rand_corr) if rand_corr is not None else np.nan,
    }
