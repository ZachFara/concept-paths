import numpy as np

from src import config as cfg
from src.data import Sample
from src import metrics


def _toy_samples():
    samples = []
    # two templates, two levels, one token each
    for t in range(2):
        for level in range(2):
            samples.append(
                Sample(
                    sample_id=f"s{t}{level}",
                    concept_name="sentiment",
                    level=level,
                    level_label=str(level),
                    template_id=t,
                    template_text="{w}",
                    synonym=f"w{t}{level}",
                    prompt_text=f"prompt {t} {level}",
                    metadata={},
                )
            )
    return samples


def test_compute_deltas_shape():
    samples = _toy_samples()
    residual = np.random.randn(len(samples), 3, 4).astype(np.float32)
    deltas = metrics.compute_deltas(samples, residual)
    assert deltas.shape[1:] == (3, 4)


def test_pca_and_rotation_shapes():
    samples = _toy_samples()
    residual = np.random.randn(len(samples), 2, 3).astype(np.float32)
    deltas = metrics.compute_deltas(samples, residual)
    pc1, k_curves, bases = metrics.compute_pca_metrics(deltas)
    assert pc1.shape[0] == residual.shape[1]
    rot = metrics.compute_rotation_metrics(bases, k_mode="fixed", k_fixed=1, k90=None)
    assert rot.shape[0] == residual.shape[1] - 1
