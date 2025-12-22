from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge


SelectionMethod = Literal["variance", "mean_abs", "probe_weight"]


def select_neurons(
    method: SelectionMethod,
    mlp_acts: np.ndarray,
    labels: np.ndarray,
    *,
    m: int,
    seed: int,
) -> np.ndarray:
    """
    Select neuron indices from MLP activations.

    mlp_acts: [n_samples, d]
    labels: [n_samples] ordinal labels
    returns: [m] indices
    """
    if mlp_acts.ndim != 2:
        raise ValueError(f"Expected mlp_acts [n_samples, d], got {mlp_acts.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels [n_samples], got {labels.shape}")
    if mlp_acts.shape[0] != labels.shape[0]:
        raise ValueError("mlp_acts and labels must align on samples")
    if m <= 0 or m > mlp_acts.shape[1]:
        raise ValueError(f"m must be in [1, d], got {m}")

    rng = np.random.default_rng(seed)
    d = mlp_acts.shape[1]
    scores: np.ndarray

    method = method.lower()
    if method == "variance":
        scores = mlp_acts.var(axis=0)
    elif method == "mean_abs":
        scores = np.mean(np.abs(mlp_acts), axis=0)
    elif method == "probe_weight":
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(mlp_acts, labels)
        scores = np.abs(model.coef_)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    jitter = rng.random(d)
    order = np.lexsort((jitter, -scores))
    return order[:m].astype(np.int64)
