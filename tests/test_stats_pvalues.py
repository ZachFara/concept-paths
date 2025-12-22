import numpy as np

from src.stats import empirical_p_two_tailed


def test_empirical_p_two_tailed_extremes() -> None:
    rng = np.random.default_rng(0)
    null = rng.normal(size=100)
    obs_hi = float(null.max() + 5.0)
    obs_lo = float(null.min() - 5.0)
    p_hi = empirical_p_two_tailed(null, obs_hi, correction=True)
    p_lo = empirical_p_two_tailed(null, obs_lo, correction=True)
    assert 0.0 < p_hi < 0.05
    assert 0.0 < p_lo < 0.05


def test_empirical_p_two_tailed_center() -> None:
    rng = np.random.default_rng(1)
    null = rng.normal(size=101)
    obs = float(np.mean(null))
    p = empirical_p_two_tailed(null, obs, correction=True)
    assert 0.2 < p <= 1.0


def test_empirical_p_two_tailed_never_zero_with_correction() -> None:
    null = np.array([0.0, 1.0, 2.0, 3.0])
    obs = 10.0
    p = empirical_p_two_tailed(null, obs, correction=True)
    assert 0.0 < p < 1.0
