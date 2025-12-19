from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .stimuli import DeltaPair


@dataclass(frozen=True)
class PCAMetrics:
    top_pc_ratio: np.ndarray  # [L]
    k90: np.ndarray  # [L], int
    sign_flips: np.ndarray  # [L], bool


def deltas_from_pairs(
    *,
    keys: list[str],
    acts: np.ndarray,
    pairs: list[DeltaPair],
) -> np.ndarray:
    key_to_row = {k: i for i, k in enumerate(keys)}
    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.zeros((0,) + acts.shape[1:], dtype=np.float32)
    neg_rows = np.array([key_to_row[p.neg_key] for p in pairs], dtype=np.int64)
    pos_rows = np.array([key_to_row[p.pos_key] for p in pairs], dtype=np.int64)
    return (acts[pos_rows] - acts[neg_rows]).astype(np.float32)


def anchor_pc1(component: np.ndarray, projections: np.ndarray, ordinal: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Flip PC1 sign so that correlation between projections and ordinal is positive.
    """
    corr = np.corrcoef(projections.flatten(), ordinal.flatten())[0, 1]
    flip = corr < 0
    return (-component if flip else component), flip


def pca_metrics_by_layer(
    deltas: np.ndarray,
    *,
    solver: Literal["full", "randomized"] = "full",
    variance_threshold: float = 0.90,
    ordinal: Optional[np.ndarray] = None,
    anchor: bool = True,
) -> PCAMetrics:
    if deltas.ndim != 3:
        raise ValueError(f"Expected deltas [n_pairs, n_layers, hidden], got {deltas.shape}")

    n_pairs, n_layers, hidden = deltas.shape
    top_pc = np.zeros((n_layers,), dtype=np.float64)
    k90 = np.zeros((n_layers,), dtype=np.int64)
    sign_flips = np.zeros((n_layers,), dtype=bool)

    for layer in range(n_layers):
        x = deltas[:, layer, :]
        if x.shape[0] < 2:
            top_pc[layer] = np.nan
            k90[layer] = 0
            continue
        pca = PCA(svd_solver=solver, whiten=False)
        pca.fit(x)
        evr = pca.explained_variance_ratio_
        comp = pca.components_[0]
        proj = x @ comp
        if anchor and ordinal is not None and ordinal.size == proj.size:
            comp, flipped = anchor_pc1(comp, proj, ordinal)
            sign_flips[layer] = flipped
        top_pc[layer] = float(evr[0]) if evr.size else np.nan
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, variance_threshold, side="left") + 1)
        k90[layer] = min(k, evr.size)

    return PCAMetrics(top_pc_ratio=top_pc.astype(np.float32), k90=k90, sign_flips=sign_flips)


def _layer_pca_basis(
    x: np.ndarray,
    *,
    k: int,
    solver: Literal["full", "randomized"] = "full",
) -> np.ndarray:
    pca = PCA(n_components=min(k, x.shape[0], x.shape[1]), svd_solver=solver, whiten=False)
    pca.fit(x)
    comps = pca.components_
    return comps[:k].T


def rotation_by_layer(
    deltas: np.ndarray,
    *,
    solver: Literal["full", "randomized"] = "full",
    k_mode: Literal["fixed", "min10_k90"] = "min10_k90",
    k_fixed: int = 5,
    k90: np.ndarray | None = None,
    metric: Literal["mean_deg", "sum_deg"] = "mean_deg",
) -> np.ndarray:
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
        u_l = _layer_pca_basis(x_l, k=k, solver=solver)
        u_lp1 = _layer_pca_basis(x_lp1, k=k, solver=solver)
        angles = subspace_angles(u_l, u_lp1)
        angles_deg = angles * (180.0 / np.pi)
        rot[layer] = float(np.mean(angles_deg) if metric == "mean_deg" else np.sum(angles_deg))
    return rot.astype(np.float32)


def random_direction_baseline(
    deltas: np.ndarray, *, n: int, rng: np.random.Generator
) -> np.ndarray:
    n_pairs, n_layers, hidden = deltas.shape
    baseline = np.zeros((n, n_layers), dtype=np.float32)
    for i in range(n):
        rand_dir = rng.normal(size=(n_layers, hidden)).astype(np.float32)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir, axis=1, keepdims=True) + 1e-8)
        proj = np.einsum("plh,lh->pl", deltas, rand_dir)
        baseline[i] = np.mean(np.abs(proj), axis=0)
    return baseline


def random_subspace_rotation(
    deltas: np.ndarray, *, n: int, k: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate random subspaces of dim k[layer] and compute rotation between them.
    """
    n_pairs, n_layers, hidden = deltas.shape
    out = np.zeros((n, n_layers - 1), dtype=np.float32)
    for i in range(n):
        bases = []
        for layer in range(n_layers):
            k_layer = max(1, int(min(k[layer], hidden)))
            rand = rng.normal(size=(hidden, k_layer))
            q, _ = np.linalg.qr(rand)
            bases.append(q[:, :k_layer])
        for layer in range(n_layers - 1):
            angles = subspace_angles(bases[layer], bases[layer + 1])
            out[i, layer] = float(np.mean(angles * (180.0 / np.pi)))
    return out


def ridge_coeffs(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model.coef_
