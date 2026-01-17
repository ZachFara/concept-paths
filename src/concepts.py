"""
The purpose of this script is to prove that we can identify concepts at each layer
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **_kwargs):
        return iterable

from src.capture import GPT2
from src.deltas import Deltas
from src.templates import SENTIMENT_SENTENCES, SENTIMENT_WORDS, Template, NULL_WORDS
from src.pca import PCA
from src.logs import setup_logger
from src.config import Config

logger = setup_logger(__name__)

class Concept:

    def __init__(self, deltas, PCA, train_test_split = .8, config = None, n_components = 1):
        self.deltas = deltas
        self.pca = PCA
        self.n_components = n_components
        self.train_split = float(train_test_split)
        if config is not None:
            self.seed = config.get("random_seed", 0)
        else:
            self.seed = 0

    @staticmethod
    def _stack_vectors(series: Iterable) -> np.ndarray:
        vecs = []
        for v in series:
            if isinstance(v, str):
                v = v.strip()
                if v.startswith("tensor(") and v.endswith(")"):
                    inner = v[len("tensor("):-1]
                    if ", dtype=" in inner:
                        inner = inner.split(", dtype=")[0]
                    inner = inner.strip()
                    if inner.startswith("[") and inner.endswith("]"):
                        inner = inner[1:-1]
                    v = np.fromstring(inner, sep=",")
                else:
                    v = np.fromstring(v, sep=",")
            elif hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            else:
                v = np.asarray(v)
            vecs.append(v)
        return np.stack(vecs, axis=0)

    def _pca_basis(self, series: Iterable, k=1) -> np.ndarray:
        X = self._stack_vectors(series)
        pca = SklearnPCA(n_components=min(k, X.shape[0]))
        pca.fit(X)
        basis = pca.components_[:k]
        norms = np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8
        return (basis / norms).astype(np.float32)

    @staticmethod
    def _subspace_similarity(U: np.ndarray, V: np.ndarray) -> float:
        Uc = U.T
        Vc = V.T
        s = np.linalg.svd(Uc.T @ Vc, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        return float(np.mean(s))

    def build_pca(self, deltas = None, n_components = None):
        if deltas is none:
            deltas = self.deltas
        if n_components is None:
            n_components = self.n_components

        pca = PCA(deltas, n)
        pca_dict = pca.get_all_layer_pca(deltas, n_components = n_components)

        return pca_dict

    def train_test_split(self, deltas = None):
        if deltas is None:
            deltas = self.deltas
        unit_col = "sentence_id"
        sentence_ids = np.array(sorted(deltas[unit_col].unique()))
        rng = np.random.default_rng(int(self.seed))
        rng.shuffle(sentence_ids)
        split_idx = int(round(len(sentence_ids) * self.train_split))
        train_ids = set(sentence_ids[:split_idx])
        train_df = deltas[deltas[unit_col].isin(train_ids)].copy()
        test_df = deltas[~deltas[unit_col].isin(train_ids)].copy()
        return train_df, test_df

    def recoverability_curve(
        self,
        train_df,
        test_df,
        k=1,
        layer_col="layer",
        delta_col="delta",
    ):
        layers = sorted(set(train_df[layer_col].unique()) & set(test_df[layer_col].unique()))
        rows = []
        for layer in layers:
            train_layer = train_df[train_df[layer_col] == layer]
            test_layer = test_df[test_df[layer_col] == layer]
            if train_layer.empty or test_layer.empty:
                continue
            train_basis = self._pca_basis(train_layer[delta_col], k=k)
            test_basis = self._pca_basis(test_layer[delta_col], k=k)
            if k == 1:
                sim = float(abs(np.dot(train_basis[0], test_basis[0])))
            else:
                sim = self._subspace_similarity(train_basis, test_basis)
            rows.append({"layer": layer, "recoverability": sim})
        return pd.DataFrame(rows)



def main():
    config = Config("config/standard.yaml")

    # Load our sentiment deltas without any recomputation
    deltas = Deltas(None)
    sentiment_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_deltas.csv")
    deltas.df = sentiment_deltas_df

    concept = Concept(None, PCA)
    sentiment_train, sentiment_test = concept.train_test_split(sentiment_deltas_df) 

    sentiment_null_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_null_deltas.csv")
    sentiment_null_train, sentiment_null_train = concept.train_test_split(sentiment_null_deltas_df)

    sentiment_recovery_df = concept.recoverability_curve(train_df=sentiment_train, test_df=sentiment_test, delta_col="delta", k=1, layer_col="layer") 

    print(sentiment_recovery_df)
    
if __name__ == "__main__":
    main()
