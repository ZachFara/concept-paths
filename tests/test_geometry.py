import numpy as np
import pytest

from src.geometry import pca_metrics_by_layer, rotation_by_layer
from src.stimuli import prompt_hash_manifest


def test_pc1_sign_anchor():
    # Construct deltas where PC1 would align negatively unless anchored.
    # Two layers, hidden=2, pairs=4
    deltas = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[2.0, 0.0], [2.0, 0.0]],
            [[3.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    ordinal = np.array([1, 2, 3, 4], dtype=np.float32)
    metrics = pca_metrics_by_layer(deltas, ordinal=ordinal, anchor=True)
    # No flips expected because correlation is positive
    assert metrics.sign_flips.shape[0] == deltas.shape[1]
    assert not metrics.sign_flips.any()


def test_prompt_hash_disjointness():
    prompts_a = ["Hello {i}".format(i=i) for i in range(3)]
    prompts_b = ["Other {i}".format(i=i) for i in range(3)]
    manifest = prompt_hash_manifest(prompts_a, prompts_b)
    assert len(manifest["discovery"]) == 3
    assert len(manifest["eval"]) == 3
    with pytest.raises(ValueError):
        prompt_hash_manifest(prompts_a, prompts_a)


def test_rotation_shape():
    deltas = np.random.randn(5, 3, 4).astype(np.float32)
    k90 = np.array([2, 2, 2])
    rot = rotation_by_layer(deltas, k90=k90)
    assert rot.shape == (deltas.shape[1] - 1,)
