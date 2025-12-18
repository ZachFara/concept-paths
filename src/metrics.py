from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

from .data import DeltaPair


@dataclass(frozen=True)
class PCAMetrics:
    top_pc_ratio: np.ndarray  # [n_layers]
    k90: np.ndarray  # [n_layers] int


def deltas_from_pairs(
    *,
    keys: list[str],
    acts: np.ndarray,
    pairs: list[DeltaPair],
) -> np.ndarray:
    """
    Convert paired activations into Δ vectors per layer.

    acts: [n_samples, n_layers, hidden]
    returns: [n_pairs, n_layers, hidden] where Δ = h(pos) - h(neg)
    """
    key_to_row = {k: i for i, k in enumerate(keys)}
    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.zeros((0,) + acts.shape[1:], dtype=np.float32)

    neg_rows = np.array([key_to_row[p.neg_key] for p in pairs], dtype=np.int64)
    pos_rows = np.array([key_to_row[p.pos_key] for p in pairs], dtype=np.int64)
    return (acts[pos_rows] - acts[neg_rows]).astype(np.float32)


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

