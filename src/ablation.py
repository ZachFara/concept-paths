from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA

from .capture import (
    build_or_load_activation_cache,
    dataset_from_samples,
    ensure_scanned,
    load_model_bundle,
    resolve_adapter,
    resolve_device,
)
from .metrics import compute_deltas
from .plotting import plot_curve_with_ci, plot_overlay_curves
from .selection import select_neurons
from .utils import ensure_dir, save_json


@dataclass(frozen=True)
class AblationResult:
    effects: Dict[int, np.ndarray]
    random_effects: Dict[int, np.ndarray]
    layer_control_effects: Dict[int, np.ndarray]
    summary: Dict[str, Any]


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def _pool_last_token(
    h: torch.Tensor,
    *,
    last_idx: torch.Tensor,
    batch_arange: torch.Tensor,
) -> torch.Tensor:
    return h[batch_arange, last_idx]


def _ensure_bsh(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    if t.ndim == 3 and t.shape[0] != batch_size and t.shape[1] == batch_size:
        return t.transpose(0, 1)
    return t


def _ensure_bsd(
    t: torch.Tensor, *, batch_size: int, seq_len: int
) -> torch.Tensor:
    if t.ndim == 3:
        return _ensure_bsh(t, batch_size)
    if t.ndim == 2:
        if t.shape[0] == batch_size * seq_len:
            return t.reshape(batch_size, seq_len, -1)
        if t.shape[0] == seq_len and batch_size == 1:
            return t.unsqueeze(0)
        if t.shape[0] == batch_size and t.shape[1] != seq_len:
            return t.unsqueeze(1)
    raise ValueError(
        f"Expected MLP activation shape [B,S,D] or flattened, got {t.shape}"
    )


def capture_residual_with_ablation(
    bundle,
    prompts: Sequence[str],
    *,
    ablate_layer: int,
    neuron_idx: np.ndarray,
    capture_layer: int,
    batch_size: int,
    mode: Literal["zero"] = "zero",
) -> np.ndarray:
    if capture_layer < ablate_layer:
        raise ValueError("capture_layer must be >= ablate_layer")
    adapter = resolve_adapter(bundle.lm, model_name=bundle.model_name)
    blocks = adapter.blocks(bundle.lm)
    if ablate_layer >= len(blocks) or capture_layer >= len(blocks):
        raise ValueError("layer index out of range")

    tokenizer = bundle.tokenizer
    if prompts:
        ensure_scanned(bundle.lm, prompts[0])

    out_chunks: list[np.ndarray] = []
    idx_t = torch.as_tensor(neuron_idx, dtype=torch.long)

    for start in range(0, len(prompts), batch_size):
        batch_prompts = list(prompts[start : start + batch_size])
        device = resolve_device(bundle.lm, bundle.device)
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        last_idx = attention_mask.sum(dim=1) - 1
        bsz = enc["input_ids"].shape[0]
        batch_arange = torch.arange(bsz, device=device)

        with bundle.lm.trace(enc):
            for layer, block in enumerate(blocks):
                if layer == ablate_layer:
                    a = adapter.ffn_out(block)
                    a_t = a.value if hasattr(a, "value") else a
                    a_t = _ensure_bsd(a_t, batch_size=bsz, seq_len=enc["input_ids"].shape[1])
                    if mode == "zero":
                        a_t[:, :, idx_t.to(a_t.device)] = 0
                    else:
                        raise ValueError(f"Unsupported ablation mode: {mode}")
                if layer == capture_layer:
                    h = adapter.block_out(block).save()

        h_t = h.value if hasattr(h, "value") else h
        h_t = _ensure_bsh(h_t, bsz)
        pooled = _pool_last_token(h_t, last_idx=last_idx, batch_arange=batch_arange)
        out_chunks.append(pooled.detach().to("cpu", dtype=torch.float32).numpy())

    return np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0, 0), dtype=np.float32)


def _concept_direction(
    samples,
    residual: np.ndarray,
    *,
    layer: int,
) -> np.ndarray:
    deltas = compute_deltas(samples, residual, method="adjacent")
    if deltas.shape[0] < 2:
        raise ValueError("Not enough delta samples to compute direction")
    pca = PCA(n_components=1, svd_solver="full", whiten=False)
    pca.fit(deltas[:, layer, :])
    return _unit(pca.components_[0].astype(np.float32))


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, rng: np.random.Generator, alpha: float) -> tuple[float, float]:
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    boots = np.zeros((n_bootstrap,), dtype=np.float32)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(values[idx]))
    low = float(np.percentile(boots, 100 * (alpha / 2.0)))
    high = float(np.percentile(boots, 100 * (1.0 - alpha / 2.0)))
    return low, high


def _sign_flip_pvalue(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    pos = np.mean(values >= 0)
    neg = np.mean(values <= 0)
    return float(2.0 * min(pos, neg))


def _random_layer_with_dim(
    rng: np.random.Generator,
    dims: Sequence[int],
    target_dim: int,
    exclude: int,
    max_layer: int,
) -> int:
    candidates = [
        i for i, d in enumerate(dims) if d == target_dim and i != exclude and i <= max_layer
    ]
    if not candidates:
        return exclude
    return int(rng.choice(candidates))


def run_ablation(
    samples_eval,
    *,
    model_name: str,
    selection_method: Literal["variance", "mean_abs", "probe_weight"] = "variance",
    layer: int,
    m_list: Sequence[int],
    direction: np.ndarray | None = None,
    alpha: float = 0.05,
    random_control: bool = True,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    artifacts_dir: Path = Path("artifacts"),
    verbose: bool = False,
) -> AblationResult:
    if not samples_eval:
        raise ValueError("samples_eval required")

    bundle = load_model_bundle(
        model_name,
        device=torch.device(device) if device else None,
    )
    dataset = dataset_from_samples(samples_eval)
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=artifacts_dir,
        batch_size=batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=True,
        verbose=verbose,
    )

    residual = cache.residual
    mlp = cache.mlp
    if layer < 0 or layer >= residual.shape[0]:
        raise ValueError("layer out of range")
    capture_layer = min(layer + 1, residual.shape[0] - 1)
    if direction is None:
        direction = _concept_direction(samples_eval, residual, layer=layer)

    direction = direction.astype(np.float32)
    direction = _unit(direction)

    labels = np.array([int(s.metadata.get("level_id", -1)) for s in samples_eval], dtype=np.float32)
    if (labels < 0).any():
        raise ValueError("Sample metadata missing level_id")

    mlp_layer = mlp[layer]
    if mlp_layer.ndim != 2:
        raise ValueError("Expected mlp layer [samples, dim]")

    baseline_proj = residual[capture_layer] @ direction

    rng = np.random.default_rng(seed)
    effects: Dict[int, np.ndarray] = {}
    random_effects: Dict[int, np.ndarray] = {}
    layer_effects: Dict[int, np.ndarray] = {}
    summary: Dict[str, Any] = {
        "layer": layer,
        "capture_layer": capture_layer,
        "selection_method": selection_method,
        "m_list": list(m_list),
        "alpha": alpha,
        "cache_key": cache.metadata.get("cache_key"),
        "dataset_signature": dataset.dataset_signature,
    }

    mlp_dims = [mlp[l].shape[1] for l in range(mlp.shape[0])]

    if verbose:
        from tqdm import tqdm

        m_iter = tqdm(m_list, desc="ablation m_list", disable=not verbose)
    else:
        m_iter = m_list
    for m in m_iter:
        idx = select_neurons(selection_method, mlp_layer, labels, m=m, seed=seed + m)
        ablated_resid = capture_residual_with_ablation(
            bundle,
            dataset.prompts,
            ablate_layer=layer,
            neuron_idx=idx,
            capture_layer=capture_layer,
            batch_size=batch_size,
        )
        ablated_proj = ablated_resid @ direction
        effect = baseline_proj - ablated_proj
        effects[m] = effect.astype(np.float32)

        if random_control:
            rand_idx = rng.choice(mlp_layer.shape[1], size=m, replace=False)
            rand_resid = capture_residual_with_ablation(
                bundle,
                dataset.prompts,
                ablate_layer=layer,
                neuron_idx=rand_idx,
                capture_layer=capture_layer,
                batch_size=batch_size,
            )
            rand_proj = rand_resid @ direction
            random_effects[m] = (baseline_proj - rand_proj).astype(np.float32)

        rand_layer = _random_layer_with_dim(
            rng,
            mlp_dims,
            mlp_layer.shape[1],
            exclude=layer,
            max_layer=capture_layer,
        )
        layer_resid = capture_residual_with_ablation(
            bundle,
            dataset.prompts,
            ablate_layer=rand_layer,
            neuron_idx=idx,
            capture_layer=capture_layer,
            batch_size=batch_size,
        )
        layer_proj = layer_resid @ direction
        layer_effects[m] = (baseline_proj - layer_proj).astype(np.float32)

    rng = np.random.default_rng(seed)
    effect_means = []
    effect_ci_low = []
    effect_ci_high = []
    p_values = []
    for m in m_list:
        vals = effects[m]
        effect_means.append(float(np.mean(vals)))
        low, high = _bootstrap_ci(vals, n_bootstrap=200, rng=rng, alpha=alpha)
        effect_ci_low.append(low)
        effect_ci_high.append(high)
        p_values.append(_sign_flip_pvalue(vals))

    auc = float(np.trapz(effect_means, x=np.array(m_list, dtype=np.float32)))
    summary.update(
        {
            "effect_means": effect_means,
            "ci_low": effect_ci_low,
            "ci_high": effect_ci_high,
            "p_values": p_values,
            "auc": auc,
        }
    )

    return AblationResult(
        effects=effects,
        random_effects=random_effects,
        layer_control_effects=layer_effects,
        summary=summary,
    )


def save_ablation_artifacts(
    result: AblationResult,
    *,
    out_dir: Path,
    stem: str,
) -> None:
    ensure_dir(out_dir)
    effects_npz = {}
    for m, vals in result.effects.items():
        effects_npz[f"effect_m{m}"] = vals
    for m, vals in result.random_effects.items():
        effects_npz[f"random_effect_m{m}"] = vals
    for m, vals in result.layer_control_effects.items():
        effects_npz[f"layer_effect_m{m}"] = vals
    np.savez_compressed(out_dir / f"{stem}_effects.npz", **effects_npz)
    save_json(out_dir / f"{stem}_summary.json", result.summary)


def plot_ablation_curves(
    result: AblationResult,
    *,
    out_dir: Path,
    stem: str,
) -> None:
    ensure_dir(out_dir)
    m_list = np.array(result.summary["m_list"], dtype=np.float32)
    mean = np.array(result.summary["effect_means"], dtype=np.float32)
    low = np.array(result.summary["ci_low"], dtype=np.float32)
    high = np.array(result.summary["ci_high"], dtype=np.float32)
    plot_curve_with_ci(
        x=m_list,
        mean=mean,
        low=low,
        high=high,
        label="effect",
        title="Ablation effect vs m",
        ylabel="Projection delta",
        outpath=out_dir / f"{stem}_effect_vs_m.png",
    )

    if result.random_effects:
        rand_mean = np.array([
            float(np.mean(result.random_effects[m])) for m in result.summary["m_list"]
        ])
        plot_overlay_curves(
            x=m_list,
            curves={"target": mean, "random": rand_mean},
            title="Target vs random control",
            ylabel="Projection delta",
            outpath=out_dir / f"{stem}_target_vs_random.png",
        )


def run_ablation_layer_sweep(
    samples_eval,
    *,
    model_name: str,
    selection_method: Literal["variance", "mean_abs", "probe_weight"] = "variance",
    m: int,
    alpha: float = 0.05,
    batch_size: int = 8,
    seed: int = 0,
    device: str | None = None,
    artifacts_dir: Path = Path("artifacts"),
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    bundle = load_model_bundle(
        model_name,
        device=torch.device(device) if device else None,
    )
    dataset = dataset_from_samples(samples_eval)
    cache = build_or_load_activation_cache(
        bundle,
        dataset=dataset,
        artifacts_dir=artifacts_dir,
        batch_size=batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=True,
        verbose=verbose,
    )
    residual = cache.residual
    mlp = cache.mlp
    labels = np.array([int(s.metadata.get("level_id", -1)) for s in samples_eval], dtype=np.float32)
    if (labels < 0).any():
        raise ValueError("Sample metadata missing level_id")

    n_layers = residual.shape[0]
    effects = np.zeros((n_layers,), dtype=np.float32)

    layer_iter = range(n_layers)
    if verbose:
        from tqdm import tqdm

        layer_iter = tqdm(layer_iter, desc="ablation layer sweep", disable=not verbose)
    for layer in layer_iter:
        capture_layer = min(layer + 1, n_layers - 1)
        direction = _concept_direction(samples_eval, residual, layer=layer)
        mlp_layer = mlp[layer]
        idx = select_neurons(selection_method, mlp_layer, labels, m=m, seed=seed + layer)
        ablated_resid = capture_residual_with_ablation(
            bundle,
            dataset.prompts,
            ablate_layer=layer,
            neuron_idx=idx,
            capture_layer=capture_layer,
            batch_size=batch_size,
        )
        baseline_proj = residual[capture_layer] @ direction
        ablated_proj = ablated_resid @ direction
        effects[layer] = float(np.mean(baseline_proj - ablated_proj))

    return {"layer_effects": effects}
