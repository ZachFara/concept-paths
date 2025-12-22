import numpy as np

from src.geometry import deltas_from_pairs
from src.stimuli import DeltaPair


def test_deltas_contract_shape() -> None:
    keys = ["a", "b", "c", "d"]
    n_layers = 4
    d_model = 6
    acts = np.arange(n_layers * len(keys) * d_model, dtype=np.float32).reshape(
        n_layers, len(keys), d_model
    )
    pairs = [
        DeltaPair(
            split="discovery",
            family="f",
            template_id=0,
            neg_level_id=0,
            pos_level_id=1,
            neg_word="x",
            pos_word="y",
            neg_key="a",
            pos_key="b",
        ),
        DeltaPair(
            split="discovery",
            family="f",
            template_id=1,
            neg_level_id=1,
            pos_level_id=2,
            neg_word="y",
            pos_word="z",
            neg_key="c",
            pos_key="d",
        ),
    ]
    deltas = deltas_from_pairs(keys=keys, acts=acts, pairs=pairs)
    assert deltas.shape == (2, n_layers, d_model)
