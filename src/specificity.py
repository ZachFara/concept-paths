from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from .capture import build_or_load_activation_cache, dataset_from_samples, load_model_bundle
from .metrics import compute_deltas, compute_pca_metrics
from .selection import select_neurons
from .ablation import capture_residual_with_ablation
from .stats import empirical_p_stats


@dataclass(frozen=True)
class SimilarityCurves:
    cosine: np.ndarray
    angles_deg: np.ndarray


@dataclass(frozen=True)
class TransferCurves:
    spearman: np.ndarray
    auc_extremes: np.ndarray


def _pc1_directions(deltas: np.ndarray) -> np.ndarray:
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas [n_pairs, layers, hidden], got {deltas.shape}")
    n_layers = deltas.shape[1]
    hidden = deltas.shape[2]
    out = np.zeros((n_layers, hidden), dtype=np.float32)
    for layer in range(n_layers):
        x = deltas[:, layer, :]
        if x.shape[0] < 2:
            out[layer] = np.nan
            continue
        pca = PCA(n_components=1, svd_solver="full", whiten=False)
        pca.fit(x)
        out[layer] = pca.components_[0].astype(np.float32)
    return out


def pc1_directions_from_samples(samples, residual: np.ndarray) -> np.ndarray:
    deltas = compute_deltas(samples, residual, method="adjacent")
    return _pc1_directions(deltas)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return (num / (denom + 1e-12)).astype(np.float32)


def similarity_across_concepts(
    samples_a,
    residual_a: np.ndarray,
    samples_b,
    residual_b: np.ndarray,
) -> SimilarityCurves:
    deltas_a = compute_deltas(samples_a, residual_a, method="adjacent")
    deltas_b = compute_deltas(samples_b, residual_b, method="adjacent")
    dirs_a = _pc1_directions(deltas_a)
    dirs_b = _pc1_directions(deltas_b)
    cosine = _cosine_similarity(dirs_a, dirs_b)

    pca_a = compute_pca_metrics(deltas_a)
    pca_b = compute_pca_metrics(deltas_b)
    angles = []
    for ua, ub in zip(pca_a.subspaces, pca_b.subspaces, strict=True):
        if ua.size == 0 or ub.size == 0:
            angles.append(np.nan)
            continue
        k = min(ua.shape[1], ub.shape[1])
        if k < 1:
            angles.append(np.nan)
            continue
        ang = subspace_angles(ua[:, :k], ub[:, :k])
        angles.append(float(np.mean(ang) * (180.0 / np.pi)))
    return SimilarityCurves(cosine=cosine, angles_deg=np.array(angles, dtype=np.float32))


def _permute_samples(samples, seed: int, levels: Sequence[str]):
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


def permutation_test_similarity(
    samples_a,
    residual_a: np.ndarray,
    samples_b,
    residual_b: np.ndarray,
    *,
    levels_a: Sequence[str],
    levels_b: Sequence[str],
    n_shuffles: int,
    seed: int,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = similarity_across_concepts(samples_a, residual_a, samples_b, residual_b)
    null_cos = np.zeros((n_shuffles, base.cosine.shape[0]), dtype=np.float32)
    null_ang = np.zeros((n_shuffles, base.angles_deg.shape[0]), dtype=np.float32)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(range(n_shuffles), desc="permute similarity", disable=False)
    else:
        iterator = range(n_shuffles)
    for i in iterator:
        perm_a = _permute_samples(samples_a, int(rng.integers(0, 1_000_000)), levels_a)
        perm_b = _permute_samples(samples_b, int(rng.integers(0, 1_000_000)), levels_b)
        sim = similarity_across_concepts(perm_a, residual_a, perm_b, residual_b)
        null_cos[i] = sim.cosine
        null_ang[i] = sim.angles_deg
    p_cos = np.zeros_like(base.cosine, dtype=np.float32)
    p_ang = np.zeros_like(base.angles_deg, dtype=np.float32)
    p_cos_hi = np.zeros_like(base.cosine, dtype=np.float32)
    p_cos_lo = np.zeros_like(base.cosine, dtype=np.float32)
    p_ang_hi = np.zeros_like(base.angles_deg, dtype=np.float32)
    p_ang_lo = np.zeros_like(base.angles_deg, dtype=np.float32)
    cos_effect = np.zeros_like(base.cosine, dtype=np.float32)
    ang_effect = np.zeros_like(base.angles_deg, dtype=np.float32)
    cos_z = np.zeros_like(base.cosine, dtype=np.float32)
    ang_z = np.zeros_like(base.angles_deg, dtype=np.float32)
    for layer in range(base.cosine.shape[0]):
        cos_stats = empirical_p_stats(null_cos[:, layer], float(base.cosine[layer]))
        ang_stats = empirical_p_stats(null_ang[:, layer], float(base.angles_deg[layer]))
        p_cos[layer] = cos_stats["p_two_tailed"]
        p_ang[layer] = ang_stats["p_two_tailed"]
        p_cos_hi[layer] = cos_stats["p_hi"]
        p_cos_lo[layer] = cos_stats["p_lo"]
        p_ang_hi[layer] = ang_stats["p_hi"]
        p_ang_lo[layer] = ang_stats["p_lo"]
        cos_effect[layer] = cos_stats["effect_size"]
        ang_effect[layer] = ang_stats["effect_size"]
        cos_z[layer] = cos_stats["z_like"]
        ang_z[layer] = ang_stats["z_like"]
    return {
        "cosine": base.cosine,
        "angles_deg": base.angles_deg,
        "null_cosine": null_cos,
        "null_angles_deg": null_ang,
        "p_cosine": p_cos.astype(np.float32),
        "p_angles": p_ang.astype(np.float32),
        "p_cosine_hi": p_cos_hi.astype(np.float32),
        "p_cosine_lo": p_cos_lo.astype(np.float32),
        "p_angles_hi": p_ang_hi.astype(np.float32),
        "p_angles_lo": p_ang_lo.astype(np.float32),
        "cosine_effect_size": cos_effect.astype(np.float32),
        "angles_effect_size": ang_effect.astype(np.float32),
        "cosine_z_like": cos_z.astype(np.float32),
        "angles_z_like": ang_z.astype(np.float32),
    }


def transfer_direction(
    direction_by_layer: np.ndarray,
    residual_b: np.ndarray,
    labels_b: np.ndarray,
) -> TransferCurves:
    n_layers = residual_b.shape[0]
    spears = np.zeros((n_layers,), dtype=np.float32)
    aucs = np.zeros((n_layers,), dtype=np.float32)
    min_level = labels_b.min()
    max_level = labels_b.max()
    extremes = (labels_b == min_level) | (labels_b == max_level)
    binary = (labels_b == max_level).astype(int)

    for layer in range(n_layers):
        scores = residual_b[layer] @ direction_by_layer[layer]
        corr = spearmanr(scores, labels_b).correlation
        spears[layer] = float(corr) if corr is not None else np.nan
        if extremes.sum() >= 2:
            aucs[layer] = float(roc_auc_score(binary[extremes], scores[extremes]))
        else:
            aucs[layer] = np.nan
    return TransferCurves(spearman=spears, auc_extremes=aucs)


def shared_variance_control(
    dir_a: np.ndarray,
    dir_b: np.ndarray,
    residual_b: np.ndarray,
    labels_b: np.ndarray,
) -> np.ndarray:
    n_layers = residual_b.shape[0]
    out = np.zeros((n_layers,), dtype=np.float32)
    for layer in range(n_layers):
        proj_b = residual_b[layer] @ dir_b[layer]
        proj_a = residual_b[layer] @ dir_a[layer]
        beta = np.dot(proj_a, proj_b) / (np.dot(proj_a, proj_a) + 1e-12)
        resid = proj_b - beta * proj_a
        corr = spearmanr(resid, labels_b).correlation
        out[layer] = float(corr) if corr is not None else np.nan
    return out


def cross_ablation_transfer(
    samples_a,
    samples_b,
    *,
    model_name: str,
    layer: int,
    m: int,
    selection_method: str,
    seed: int,
    batch_size: int,
    device: str | None,
    artifacts_dir,
) -> Dict[str, float]:
    bundle = load_model_bundle(
        model_name,
        device=torch.device(device) if device else None,
    )
    dataset_a = dataset_from_samples(samples_a)
    dataset_b = dataset_from_samples(samples_b)
    cache_a = build_or_load_activation_cache(
        bundle,
        dataset=dataset_a,
        artifacts_dir=artifacts_dir,
        batch_size=batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=True,
    )
    cache_b = build_or_load_activation_cache(
        bundle,
        dataset=dataset_b,
        artifacts_dir=artifacts_dir,
        batch_size=batch_size,
        pooling="last",
        capture_sites=("residual", "mlp"),
        use_cache=True,
    )

    labels_a = np.array([int(s.metadata.get("level_id", -1)) for s in samples_a], dtype=np.float32)
    labels_b = np.array([int(s.metadata.get("level_id", -1)) for s in samples_b], dtype=np.float32)
    if (labels_a < 0).any() or (labels_b < 0).any():
        raise ValueError("Sample metadata missing level_id")

    deltas_a = compute_deltas(samples_a, cache_a.residual, method="adjacent")
    deltas_b = compute_deltas(samples_b, cache_b.residual, method="adjacent")
    dir_a = _pc1_directions(deltas_a)[layer]
    dir_b = _pc1_directions(deltas_b)[layer]

    idx = select_neurons(selection_method, cache_a.mlp[layer], labels_a, m=m, seed=seed)

    capture_layer = min(layer + 1, cache_a.residual.shape[0] - 1)
    base_a = cache_a.residual[capture_layer] @ dir_a
    base_b = cache_b.residual[capture_layer] @ dir_b

    ablated_a = capture_residual_with_ablation(
        bundle,
        dataset_a.prompts,
        ablate_layer=layer,
        neuron_idx=idx,
        capture_layer=capture_layer,
        batch_size=batch_size,
    )
    ablated_b = capture_residual_with_ablation(
        bundle,
        dataset_b.prompts,
        ablate_layer=layer,
        neuron_idx=idx,
        capture_layer=capture_layer,
        batch_size=batch_size,
    )

    effect_a = float(np.mean(base_a - (ablated_a @ dir_a)))
    effect_b = float(np.mean(base_b - (ablated_b @ dir_b)))
    return {"effect_a": effect_a, "effect_b": effect_b}
