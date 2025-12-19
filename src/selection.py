from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge


def select_neurons(
    *,
    method: Literal["variance", "mean_abs", "probe", "corr"],
    mlp_acts: np.ndarray,  # [N, L, D]
    labels: np.ndarray,  # [N]
    layer: int,
    m: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Rank neurons by a criterion and return top-m indices for a layer.
    """
    rng = np.random.default_rng(seed)
    x = mlp_acts[:, layer, :]  # [N, D]
    scores: np.ndarray
    if method == "variance":
        scores = x.var(axis=0)
    elif method == "mean_abs":
        scores = np.mean(np.abs(x), axis=0)
    elif method == "corr":
        labels_c = labels - labels.mean()
        x_c = x - x.mean(axis=0, keepdims=True)
        num = (x_c * labels_c[:, None]).sum(axis=0)
        denom = np.sqrt((x_c**2).sum(axis=0) * (labels_c**2).sum() + 1e-8)
        scores = num / denom
    elif method == "probe":
        model = Ridge(alpha=1.0)
        model.fit(x, labels)
        scores = model.coef_
    else:
        raise ValueError(f"Unknown method {method}")
    order = np.argsort(-np.abs(scores))
    order = np.array(order, dtype=np.int64)
    m_eff = min(m, order.size)
    top = order[:m_eff]
    # tie-breaking deterministically by shuffling stable? ensure deterministic
    return top
