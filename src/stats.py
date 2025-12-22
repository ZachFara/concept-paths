from __future__ import annotations

from typing import Dict

import numpy as np


def empirical_p_two_tailed(
    null: np.ndarray,
    obs: float,
    *,
    correction: bool = True,
) -> float:
    null = np.asarray(null).ravel()
    n = null.size
    if n == 0:
        return float("nan")
    k_hi = int(np.sum(null >= obs))
    k_lo = int(np.sum(null <= obs))
    if correction:
        p_hi = (k_hi + 1.0) / (n + 1.0)
        p_lo = (k_lo + 1.0) / (n + 1.0)
    else:
        p_hi = k_hi / n
        p_lo = k_lo / n
    p_two = 2.0 * min(p_hi, p_lo)
    return float(min(p_two, 1.0))


def empirical_p_stats(
    null: np.ndarray,
    obs: float,
    *,
    correction: bool = True,
    eps: float = 1e-12,
) -> Dict[str, float]:
    null = np.asarray(null).ravel()
    n = null.size
    if n == 0:
        return {
            "p_two_tailed": float("nan"),
            "p_hi": float("nan"),
            "p_lo": float("nan"),
            "effect_size": float("nan"),
            "z_like": float("nan"),
        }
    k_hi = int(np.sum(null >= obs))
    k_lo = int(np.sum(null <= obs))
    if correction:
        p_hi = (k_hi + 1.0) / (n + 1.0)
        p_lo = (k_lo + 1.0) / (n + 1.0)
    else:
        p_hi = k_hi / n
        p_lo = k_lo / n
    p_two = min(1.0, 2.0 * min(p_hi, p_lo))
    mean = float(np.mean(null))
    std = float(np.std(null))
    effect = float(obs - mean)
    z_like = float(effect / (std + eps))
    return {
        "p_two_tailed": float(p_two),
        "p_hi": float(p_hi),
        "p_lo": float(p_lo),
        "effect_size": effect,
        "z_like": z_like,
    }
