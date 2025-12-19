from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

from .data import Sample, shuffle_labels_preserve_counts


def compute_deltas(
    samples: List[Sample],
    residual: np.ndarray,
    *,
    method: Literal["adjacent"] = "adjacent",
) -> np.ndarray:
    """
    Build Δ vectors per layer for adjacent levels within each template.

    residual: [N, L, H] aligned with samples order.
    returns: [P, L, H]
    """
    if residual.shape[0] != len(samples):
        raise ValueError("Residual rows must match number of samples")
    if method != "adjacent":
        raise ValueError("Only 'adjacent' method supported")
    # Group indices by (template_id, level)
    by_template: Dict[int, Dict[int, List[int]]] = {}
    for idx, s in enumerate(samples):
        by_template.setdefault(s.template_id, {}).setdefault(s.level, []).append(idx)
    pairs: List[Tuple[int, int]] = []
    for template_id, levels in by_template.items():
        max_level = max(levels.keys())
        for lvl in range(max_level):
            if lvl not in levels or (lvl + 1) not in levels:
                continue
            for i in levels[lvl]:
                for j in levels[lvl + 1]:
                    pairs.append((i, j))
    if not pairs:
        return np.zeros((0,) + residual.shape[1:], dtype=np.float32)
    neg_idx = np.array([i for i, _ in pairs], dtype=np.int64)
    pos_idx = np.array([j for _, j in pairs], dtype=np.int64)
    deltas = residual[pos_idx] - residual[neg_idx]
    return deltas.astype(np.float32)


def _pca_layer(x: np.ndarray, thresholds: List[float]) -> Tuple[float, Dict[float, int], np.ndarray]:
    """
    Run PCA on [P, H] returning top_pc ratio, k thresholds, and components (basis).
    """
    if x.shape[0] < 2:
        return np.nan, {t: 0 for t in thresholds}, np.zeros((x.shape[1], 0), dtype=np.float32)
    pca = PCA(svd_solver="full")
    pca.fit(x)
    evr = pca.explained_variance_ratio_
    top_pc = float(evr[0]) if evr.size else np.nan
    k_dict: Dict[float, int] = {}
    cum = np.cumsum(evr)
    for t in thresholds:
        k = int(np.searchsorted(cum, t, side="left") + 1)
        k_dict[t] = min(k, evr.size)
    comps = pca.components_.T  # [H, K]
    return top_pc, k_dict, comps


def compute_pca_metrics(
    deltas: np.ndarray,
    *,
    thresholds: List[float] = [0.8, 0.9, 0.95],
) -> Tuple[np.ndarray, Dict[float, np.ndarray], List[np.ndarray]]:
    """
    Returns pc1 curve, k curves per threshold, and subspace bases per layer.
    """
    if deltas.ndim != 3:
        raise ValueError("deltas must be [P, L, H]")
    _, n_layers, _ = deltas.shape
    pc1 = np.zeros((n_layers,), dtype=np.float32)
    k_curves: Dict[float, np.ndarray] = {t: np.zeros((n_layers,), dtype=np.int64) for t in thresholds}
    bases: List[np.ndarray] = []
    for l in range(n_layers):
        top_pc, k_dict, comps = _pca_layer(deltas[:, l, :], thresholds)
        pc1[l] = top_pc
        for t in thresholds:
            k_curves[t][l] = k_dict[t]
        bases.append(comps.astype(np.float32))
    return pc1, k_curves, bases


def compute_rotation_metrics(
    bases: List[np.ndarray],
    *,
    k_mode: Literal["fixed", "min10_k90"] = "min10_k90",
    k_fixed: int = 5,
    k90: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(bases) < 2:
        return np.zeros((0,), dtype=np.float32)
    rot = np.zeros((len(bases) - 1,), dtype=np.float32)
    for i in range(len(bases) - 1):
        a = bases[i]
        b = bases[i + 1]
        if a.shape[0] == 0 or b.shape[0] == 0:
            rot[i] = np.nan
            continue
        if k_mode == "fixed":
            k = min(k_fixed, a.shape[1], b.shape[1])
        else:
            if k90 is None:
                raise ValueError("k90 required for min10_k90")
            k = min(int(max(1, min(10, k90[i]))), a.shape[1], b.shape[1])
        if k < 1:
            rot[i] = np.nan
            continue
        angles = subspace_angles(a[:, :k], b[:, :k])
        rot[i] = float(np.mean(angles * (180.0 / np.pi)))
    return rot


def bootstrap_curves(
    deltas: np.ndarray,
    *,
    n_boot: int = 100,
    thresholds: List[float] = [0.8, 0.9, 0.95],
    k_mode: Literal["fixed", "min10_k90"] = "min10_k90",
    k_fixed: int = 5,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    pc1_samples = []
    k_samples = {t: [] for t in thresholds}
    rot_samples = []
    n = deltas.shape[0]
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot = deltas[idx]
        pc1, k_curves, bases = compute_pca_metrics(boot, thresholds=thresholds)
        rot = compute_rotation_metrics(bases, k_mode=k_mode, k_fixed=k_fixed, k90=k_curves[0.9])
        pc1_samples.append(pc1)
        rot_samples.append(rot)
        for t in thresholds:
            k_samples[t].append(k_curves[t])
    pc1_arr = np.stack(pc1_samples)
    rot_arr = np.stack(rot_samples)
    k_arrs = {t: np.stack(v) for t, v in k_samples.items()}
    def ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)
    out = {
        "pc1_mean": pc1_arr.mean(axis=0),
        "pc1_ci_low": ci(pc1_arr)[0],
        "pc1_ci_high": ci(pc1_arr)[1],
        "rotation_mean": rot_arr.mean(axis=0),
        "rotation_ci_low": ci(rot_arr)[0],
        "rotation_ci_high": ci(rot_arr)[1],
    }
    for t, arr in k_arrs.items():
        low, high = ci(arr)
        out[f"k{int(t*100)}_mean"] = arr.mean(axis=0)
        out[f"k{int(t*100)}_ci_low"] = low
        out[f"k{int(t*100)}_ci_high"] = high
    return out


def permutation_null(
    samples: List[Sample],
    residual: np.ndarray,
    *,
    n_shuffles: int,
    thresholds: List[float],
    early_layers: List[int],
    late_layers: List[int],
) -> Tuple[List[float], List[float]]:
    rng = np.random.default_rng(0)
    null_k_scores: List[float] = []
    null_rot_scores: List[float] = []
    levels = [s.level for s in samples]
    level_labels = [s.level_label for s in samples]
    unique_levels = sorted(set(level_labels))
    for _ in range(n_shuffles):
        shuffled_levels = shuffle_labels_preserve_counts(levels, rng)
        shuffled_samples: List[Sample] = []
        for s, new_lvl in zip(samples, shuffled_levels, strict=True):
            new_label = unique_levels[new_lvl % len(unique_levels)]
            shuffled_samples.append(
                Sample(
                    sample_id=s.sample_id,
                    concept_name=s.concept_name,
                    level=int(new_lvl),
                    level_label=new_label,
                    template_id=s.template_id,
                    template_text=s.template_text,
                    synonym=s.synonym,
                    prompt_text=s.prompt_text,
                    metadata=s.metadata,
                )
            )
        deltas = compute_deltas(shuffled_samples, residual)
        pc1, k_curves, bases = compute_pca_metrics(deltas, thresholds=thresholds)
        rot = compute_rotation_metrics(bases, k_mode="min10_k90", k90=k_curves[0.9])
        k_score, r_score = summary_scores(
            k_curves[0.9],
            rot,
            early_layers=early_layers,
            late_layers=late_layers,
        )
        null_k_scores.append(k_score)
        null_rot_scores.append(r_score)
    return null_k_scores, null_rot_scores


def summary_scores(
    k_curve: np.ndarray,
    rot_curve: np.ndarray,
    *,
    early_layers: List[int],
    late_layers: List[int],
) -> Tuple[float, float]:
    def _idx(idx_list: List[int], n: int) -> List[int]:
        out = []
        for i in idx_list:
            out.append(i if i >= 0 else n + i)
        return [i for i in out if 0 <= i < n]

    k_idx_e = _idx(early_layers, len(k_curve))
    k_idx_l = _idx(late_layers, len(k_curve))
    r_idx_e = _idx(early_layers[:-1], len(rot_curve))
    r_idx_l = _idx(late_layers[:-1], len(rot_curve))
    k_score = float(np.nanmean(k_curve[k_idx_l]) - np.nanmean(k_curve[k_idx_e])) if k_idx_e and k_idx_l else 0.0
    r_score = float(np.nanmean(rot_curve[r_idx_e]) - np.nanmean(rot_curve[r_idx_l])) if r_idx_e and r_idx_l and len(rot_curve)>0 else 0.0
    return k_score, r_score
