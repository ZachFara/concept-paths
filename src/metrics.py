from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Sequence, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

from .data import DeltaPair, Sample


@dataclass(frozen=True)
class PCAMetrics:
    top_pc_ratio: np.ndarray  # [n_layers]
    k90: np.ndarray  # [n_layers] int


@dataclass(frozen=True)
class PCACurves:
    pc1_curve: np.ndarray  # [n_layers]
    k_curves: Dict[str, np.ndarray]  # {"k80": [n_layers], ...}
    subspaces: list[np.ndarray]  # per-layer [hidden, k]


@dataclass(frozen=True)
class RotationCurves:
    rotation_curve: np.ndarray  # [n_layers - 1]


def deltas_from_pairs(
    *,
    keys: list[str],
    acts: np.ndarray,
    pairs: list[DeltaPair],
) -> np.ndarray:
    """
    Convert paired activations into Δ vectors per layer.

    acts: [layers, n_samples, hidden]
    returns: [n_pairs, layers, hidden] where Δ = h(pos) - h(neg)
    """
    if acts.ndim != 3:
        raise ValueError(f"Expected acts with 3 dims [layers, samples, hidden], got {acts.shape}")
    if acts.shape[1] != len(keys):
        raise ValueError(
            f"Expected acts.shape[1] == len(keys) ({len(keys)}), got {acts.shape}"
        )
    key_to_row = {k: i for i, k in enumerate(keys)}
    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.zeros((0, acts.shape[0], acts.shape[2]), dtype=np.float32)

    neg_rows = np.array([key_to_row[p.neg_key] for p in pairs], dtype=np.int64)
    pos_rows = np.array([key_to_row[p.pos_key] for p in pairs], dtype=np.int64)
    deltas = (acts[:, pos_rows, :] - acts[:, neg_rows, :]).astype(np.float32)
    return np.transpose(deltas, (1, 0, 2))


def compute_deltas(
    samples: Sequence[Sample],
    residual_by_layer: np.ndarray,
    *,
    method: Literal["adjacent"] = "adjacent",
    concept_mode: Literal["sentiment", "unordered", "topic_control"] = "sentiment",
    topic_pair_strategy: Literal["cartesian", "random"] = "cartesian",
    pair_subsample_frac: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute Δ vectors per layer based on adjacent ordinal levels within each template.

    residual_by_layer: [layers, n_samples, hidden]
    returns: [n_pairs, n_layers, hidden]
    """
    if residual_by_layer.ndim != 3:
        raise ValueError(
            f"Expected residual_by_layer [layers, samples, hidden], got {residual_by_layer.shape}"
        )
    if residual_by_layer.shape[1] != len(samples):
        raise ValueError(
            f"Expected residual_by_layer.shape[1] == len(samples) ({len(samples)}), "
            f"got {residual_by_layer.shape}"
        )
    if method != "adjacent":
        raise ValueError(f"Unsupported method: {method}")

    if concept_mode == "topic_control":
        return _compute_topic_control_deltas(
            samples,
            residual_by_layer,
            strategy=topic_pair_strategy,
            pair_subsample_frac=pair_subsample_frac,
            seed=seed,
        )

    levels_by_template: Dict[int, Dict[int, list[int]]] = {}
    for idx, s in enumerate(samples):
        level_id = int(s.metadata.get("level_id", -1))
        if level_id < 0:
            raise ValueError("Sample metadata missing level_id")
        levels_by_template.setdefault(s.template_id, {}).setdefault(level_id, []).append(idx)

    n_layers = residual_by_layer.shape[0]
    hidden = residual_by_layer.shape[2]
    deltas: list[np.ndarray] = []

    for template_id, by_level in levels_by_template.items():
        level_ids = sorted(by_level.keys())
        for neg_level_id, pos_level_id in zip(level_ids, level_ids[1:], strict=False):
            neg_idx = by_level[neg_level_id]
            pos_idx = by_level[pos_level_id]
            if not neg_idx or not pos_idx:
                continue
            for n in neg_idx:
                for p in pos_idx:
                    delta = residual_by_layer[:, p, :] - residual_by_layer[:, n, :]
                    deltas.append(delta.astype(np.float32))

    if not deltas:
        return np.zeros((0, n_layers, hidden), dtype=np.float32)
    return np.stack(deltas, axis=0)


def build_topic_control_pairs(
    samples: Sequence[Sample],
    *,
    strategy: Literal["cartesian", "random"] = "cartesian",
    pair_subsample_frac: float | None = None,
    seed: int = 0,
) -> list[tuple[int, int]]:
    if not samples:
        return []
    rng = np.random.default_rng(seed)
    by_key: Dict[Tuple[int, str], list[int]] = {}
    for idx, s in enumerate(samples):
        topic = s.metadata.get("topic")
        if topic is None:
            raise ValueError("Sample metadata missing topic for topic_control")
        by_key.setdefault((s.template_id, s.synonym), []).append(idx)

    pairs: list[tuple[int, int]] = []
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        if strategy == "cartesian":
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pairs.append((idxs[i], idxs[j]))
        elif strategy == "random":
            total = len(idxs) * (len(idxs) - 1) // 2
            target = min(total, len(idxs))
            seen = set()
            for _ in range(target):
                a, b = rng.choice(idxs, size=2, replace=False)
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        else:
            raise ValueError(f"Unknown topic_pair_strategy: {strategy}")

    if pair_subsample_frac is not None and 0.0 < pair_subsample_frac < 1.0 and pairs:
        k = max(1, int(len(pairs) * pair_subsample_frac))
        idx = rng.choice(len(pairs), size=k, replace=False)
        pairs = [pairs[int(i)] for i in idx]
    return pairs


def _compute_topic_control_deltas(
    samples: Sequence[Sample],
    residual_by_layer: np.ndarray,
    *,
    strategy: Literal["cartesian", "random"],
    pair_subsample_frac: float | None,
    seed: int,
) -> np.ndarray:
    pairs = build_topic_control_pairs(
        samples,
        strategy=strategy,
        pair_subsample_frac=pair_subsample_frac,
        seed=seed,
    )
    if not pairs:
        return np.zeros((0, residual_by_layer.shape[0], residual_by_layer.shape[2]), dtype=np.float32)
    deltas = []
    for neg_idx, pos_idx in pairs:
        delta = residual_by_layer[:, pos_idx, :] - residual_by_layer[:, neg_idx, :]
        deltas.append(delta.astype(np.float32))
    return np.stack(deltas, axis=0)


def pc1_directions_from_deltas(
    deltas: np.ndarray,
    *,
    solver: Literal["full", "randomized"] = "full",
) -> np.ndarray:
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
        pca = PCA(n_components=1, svd_solver=solver, whiten=False)
        pca.fit(x)
        vec = pca.components_[0].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        out[layer] = vec / (norm + 1e-12)
    return out


def angle_between_directions(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if u.shape != v.shape:
        raise ValueError(f"Expected matching shapes for directions, got {u.shape} vs {v.shape}")
    n_layers = u.shape[0]
    angles = np.zeros((n_layers,), dtype=np.float32)
    for layer in range(n_layers):
        a = u[layer]
        b = v[layer]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            angles[layer] = np.nan
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            angles[layer] = np.nan
            continue
        cos = float(np.dot(a, b) / denom)
        cos = max(-1.0, min(1.0, abs(cos)))
        angles[layer] = float(np.degrees(np.arccos(cos)))
    return angles


def explained_fraction_along_direction(
    deltas: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas [n_pairs, layers, hidden], got {deltas.shape}")
    if direction.ndim != 2:
        raise ValueError(f"Expected direction [layers, hidden], got {direction.shape}")
    if deltas.shape[1] != direction.shape[0] or deltas.shape[2] != direction.shape[1]:
        raise ValueError(
            f"Mismatched shapes deltas {deltas.shape} vs direction {direction.shape}"
        )
    n_layers = deltas.shape[1]
    out = np.zeros((n_layers,), dtype=np.float32)
    for layer in range(n_layers):
        x = deltas[:, layer, :]
        if x.shape[0] < 2:
            out[layer] = np.nan
            continue
        proj = x @ direction[layer]
        var_proj = float(np.var(proj, ddof=1))
        var_total = float(np.sum(np.var(x, axis=0, ddof=1)))
        out[layer] = float(var_proj / (var_total + 1e-12))
    return out


def compute_pca_metrics(
    deltas: np.ndarray,
    *,
    solver: Literal["full", "randomized"] = "full",
    thresholds: Sequence[float] = (0.80, 0.90, 0.95),
) -> PCACurves:
    """
    Compute PCA curves and per-layer subspaces from delta vectors.

    deltas: [n_pairs, n_layers, hidden]
    returns: pc1_curve, k_curves, subspaces per layer
    """
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas [n_pairs, n_layers, hidden], got {deltas.shape}")

    n_pairs, n_layers, hidden = deltas.shape
    pc1_curve = np.zeros((n_layers,), dtype=np.float32)
    k_curves = {f"k{int(t * 100)}": np.zeros((n_layers,), dtype=np.int64) for t in thresholds}
    subspaces: list[np.ndarray] = []

    for layer in range(n_layers):
        x = deltas[:, layer, :]
        if x.shape[0] < 2:
            pc1_curve[layer] = np.nan
            for key in k_curves:
                k_curves[key][layer] = 0
            subspaces.append(np.zeros((hidden, 0), dtype=np.float32))
            continue

        pca = PCA(svd_solver=solver, whiten=False)
        pca.fit(x)
        evr = pca.explained_variance_ratio_
        pc1_curve[layer] = float(evr[0]) if evr.size else np.nan
        cum = np.cumsum(evr)
        for t in thresholds:
            k = int(np.searchsorted(cum, t, side="left") + 1)
            k_curves[f"k{int(t * 100)}"][layer] = min(k, evr.size)
        k90 = int(k_curves["k90"][layer]) if "k90" in k_curves else 1
        k90 = max(1, min(k90, evr.size))
        subspace = pca.components_[:k90].T.astype(np.float32)
        subspaces.append(subspace)

    return PCACurves(
        pc1_curve=pc1_curve,
        k_curves={k: v.astype(np.int64) for k, v in k_curves.items()},
        subspaces=subspaces,
    )


def compute_rotation_metrics(subspaces: Sequence[np.ndarray]) -> RotationCurves:
    """
    Compute mean principal-angle rotation between adjacent layer subspaces.
    """
    n_layers = len(subspaces)
    if n_layers < 2:
        return RotationCurves(rotation_curve=np.zeros((0,), dtype=np.float32))
    rot = np.zeros((n_layers - 1,), dtype=np.float32)
    for layer in range(n_layers - 1):
        u_l = subspaces[layer]
        u_lp1 = subspaces[layer + 1]
        if u_l.size == 0 or u_lp1.size == 0:
            rot[layer] = np.nan
            continue
        k = min(u_l.shape[1], u_lp1.shape[1])
        if k < 1:
            rot[layer] = np.nan
            continue
        angles = subspace_angles(u_l[:, :k], u_lp1[:, :k])
        rot[layer] = float(np.mean(angles) * (180.0 / np.pi))
    return RotationCurves(rotation_curve=rot)


def bootstrap_curves(
    samples: Sequence[Sample],
    residual_by_layer: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    solver: Literal["full", "randomized"] = "full",
    thresholds: Sequence[float] = (0.80, 0.90, 0.95),
) -> Dict[str, np.ndarray]:
    """
    Bootstrap CI over prompts by resampling sample indices with replacement.
    Returns percentile bands for pc1, k-curves, and rotation.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError("samples required for bootstrap")

    pc1_stack: list[np.ndarray] = []
    k_stack: Dict[str, list[np.ndarray]] = {f"k{int(t * 100)}": [] for t in thresholds}
    rot_stack: list[np.ndarray] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_samples = [samples[int(i)] for i in idx]
        boot_resid = residual_by_layer[:, idx, :]
        deltas = compute_deltas(boot_samples, boot_resid, method="adjacent")
        pca = compute_pca_metrics(deltas, solver=solver, thresholds=thresholds)
        rot = compute_rotation_metrics(pca.subspaces)
        pc1_stack.append(pca.pc1_curve)
        for key in k_stack:
            k_stack[key].append(pca.k_curves[key].astype(np.float32))
        rot_stack.append(rot.rotation_curve)

    pc1_arr = np.stack(pc1_stack, axis=0)
    rot_arr = np.stack(rot_stack, axis=0) if rot_stack else np.zeros((0, 0), dtype=np.float32)
    bands = {
        "pc1_low": np.percentile(pc1_arr, 2.5, axis=0),
        "pc1_high": np.percentile(pc1_arr, 97.5, axis=0),
        "rotation_low": np.percentile(rot_arr, 2.5, axis=0) if rot_arr.size else rot_arr,
        "rotation_high": np.percentile(rot_arr, 97.5, axis=0) if rot_arr.size else rot_arr,
    }
    for key, stack in k_stack.items():
        arr = np.stack(stack, axis=0)
        bands[f"{key}_low"] = np.percentile(arr, 2.5, axis=0)
        bands[f"{key}_high"] = np.percentile(arr, 97.5, axis=0)
    return {k: v.astype(np.float32) for k, v in bands.items()}


def pca_metrics_by_layer(
    deltas: np.ndarray,
    *,
    solver: Literal["full", "randomized"] = "full",
    variance_threshold: float = 0.90,
) -> PCAMetrics:
    """
    For each layer l, run PCA on Δh_l and compute:
      - top PC explained variance ratio
      - k90: number of PCs to reach 90% cumulative variance
    """
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas [n_pairs, n_layers, hidden], got {deltas.shape}")

    n_pairs, n_layers, hidden = deltas.shape
    top_pc = np.zeros((n_layers,), dtype=np.float64)
    k90 = np.zeros((n_layers,), dtype=np.int64)

    for layer in range(n_layers):
        x = deltas[:, layer, :]  # [n_pairs, hidden]
        # sklearn requires at least 2 samples; if too small, return degenerate values.
        if x.shape[0] < 2:
            top_pc[layer] = np.nan
            k90[layer] = 0
            continue

        pca = PCA(svd_solver=solver, whiten=False)
        pca.fit(x)
        evr = pca.explained_variance_ratio_
        top_pc[layer] = float(evr[0]) if evr.size else np.nan

        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, variance_threshold, side="left") + 1)
        k90[layer] = min(k, evr.size)

    return PCAMetrics(top_pc_ratio=top_pc.astype(np.float32), k90=k90)


def _layer_pca_basis(
    x: np.ndarray,
    *,
    k: int,
    solver: Literal["full", "randomized"] = "full",
) -> np.ndarray:
    """
    Return an orthonormal basis U for the top-k PCA subspace of x.
    U has shape [hidden, k].
    """
    pca = PCA(n_components=min(k, x.shape[0], x.shape[1]), svd_solver=solver, whiten=False)
    pca.fit(x)
    comps = pca.components_  # [k, hidden] rows are PCs
    u = comps[:k].T  # [hidden, k]
    # sklearn components_ are already orthonormal (up to numerical precision)
    return u


def rotation_by_layer(
    deltas: np.ndarray,
    *,
    solver: Literal["full", "randomized"] = "full",
    k_mode: Literal["fixed", "min10_k90"] = "min10_k90",
    k_fixed: int = 5,
    k90: np.ndarray | None = None,
    metric: Literal["mean_deg", "sum_deg"] = "mean_deg",
) -> np.ndarray:
    """
    Compute subspace rotation between adjacent layers using principal angles.

    Returns rotation[l] for (layer l) -> (layer l+1); shape [n_layers - 1]
    """
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas [n_pairs, n_layers, hidden], got {deltas.shape}")

    n_pairs, n_layers, hidden = deltas.shape
    if n_layers < 2:
        return np.zeros((0,), dtype=np.float32)

    if k_mode == "min10_k90" and k90 is None:
        raise ValueError("k90 must be provided when k_mode='min10_k90'")

    rot = np.zeros((n_layers - 1,), dtype=np.float64)

    for layer in range(n_layers - 1):
        x_l = deltas[:, layer, :]
        x_lp1 = deltas[:, layer + 1, :]
        if x_l.shape[0] < 2 or x_lp1.shape[0] < 2:
            rot[layer] = np.nan
            continue

        if k_mode == "fixed":
            k = k_fixed
        else:
            k = int(min(10, int(k90[layer])))
            k = max(k, 1)

        k = int(min(k, x_l.shape[0], x_l.shape[1], x_lp1.shape[0], x_lp1.shape[1]))
        if k < 1:
            rot[layer] = np.nan
            continue

        u_l = _layer_pca_basis(x_l, k=k, solver=solver)  # [hidden, k]
        u_lp1 = _layer_pca_basis(x_lp1, k=k, solver=solver)
        angles = subspace_angles(u_l, u_lp1)  # radians, [k]
        angles_deg = angles * (180.0 / np.pi)

        rot[layer] = float(np.mean(angles_deg) if metric == "mean_deg" else np.sum(angles_deg))

    return rot.astype(np.float32)
