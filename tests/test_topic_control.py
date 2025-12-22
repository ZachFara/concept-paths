import numpy as np

from src.config import ProjectConfig
from src.data import generate_samples
from src.metrics import build_topic_control_pairs, compute_deltas, compute_pca_metrics


def test_topic_control_pairs_fix_adjective() -> None:
    cfg = ProjectConfig()
    samples = generate_samples(
        "sentiment",
        "discovery",
        "topic_swap_fixed_sentiment",
        seed=0,
        n_per_level=1,
        data_spec=cfg.data,
        control_spec=cfg.controls,
        concept_mode="topic_control",
    )
    pairs = build_topic_control_pairs(samples, strategy="cartesian", seed=0)
    assert pairs, "expected topic_control pairs"
    for i, j in pairs:
        assert samples[i].synonym == samples[j].synonym


def test_topic_control_geometry_smoke() -> None:
    cfg = ProjectConfig()
    samples = generate_samples(
        "sentiment",
        "discovery",
        "topic_swap_fixed_sentiment",
        seed=0,
        n_per_level=1,
        data_spec=cfg.data,
        control_spec=cfg.controls,
        concept_mode="topic_control",
    )
    residual = np.random.default_rng(0).normal(
        size=(2, len(samples), 4)
    ).astype(np.float32)
    deltas = compute_deltas(
        samples,
        residual,
        method="adjacent",
        concept_mode="topic_control",
        topic_pair_strategy="cartesian",
        seed=0,
    )
    pca = compute_pca_metrics(deltas)
    assert pca.pc1_curve.shape == (2,)
