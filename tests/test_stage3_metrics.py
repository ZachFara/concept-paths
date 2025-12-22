import numpy as np

from src.config import ProjectConfig
from src.data import generate_samples
from src.metrics import (
    compute_deltas,
    compute_pca_metrics,
    compute_rotation_metrics,
    bootstrap_curves,
)


def test_stage3_metric_shapes() -> None:
    cfg = ProjectConfig()
    concept = "sentiment"
    template_family = list(cfg.data.concepts[concept].templates.keys())[0]
    samples = generate_samples(
        concept,
        "discovery",
        template_family,
        seed=0,
        n_per_level=1,
        data_spec=cfg.data,
        control_spec=cfg.controls,
    )
    n_layers = 3
    hidden = 4
    residual = np.random.default_rng(0).normal(size=(n_layers, len(samples), hidden)).astype(
        np.float32
    )
    deltas = compute_deltas(samples, residual, method="adjacent")
    assert deltas.shape[1] == n_layers
    assert deltas.shape[2] == hidden

    pca = compute_pca_metrics(deltas)
    assert pca.pc1_curve.shape == (n_layers,)
    assert pca.k_curves["k80"].shape == (n_layers,)
    assert len(pca.subspaces) == n_layers

    rot = compute_rotation_metrics(pca.subspaces)
    assert rot.rotation_curve.shape == (n_layers - 1,)

    bands = bootstrap_curves(
        samples,
        residual,
        n_bootstrap=5,
        rng=np.random.default_rng(0),
    )
    assert bands["pc1_low"].shape == (n_layers,)
    assert bands["k90_high"].shape == (n_layers,)
