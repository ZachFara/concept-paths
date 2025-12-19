import numpy as np

from src import specificity


def test_cosine_similarity():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert np.isclose(specificity.cosine_similarity(a, b), 1.0)
