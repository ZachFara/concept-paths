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

    def train_test_split(self, deltas = None, seed = None):
        if deltas is None:
            deltas = self.deltas
        if seed is None:
            seed = self.seed
        unit_col = "sentence_id"
        sentence_ids = np.array(sorted(deltas[unit_col].unique()))
        rng = np.random.default_rng(int(seed))
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
            # Just compute the PCA and avoid any interaction with our PCA library
            train_basis = self._pca_basis(train_layer[delta_col], k=k)
            test_basis = self._pca_basis(test_layer[delta_col], k=k)
            if k == 1:
                sim = float(abs(np.dot(train_basis[0], test_basis[0])))
            else:
                sim = self._subspace_similarity(train_basis, test_basis)
            rows.append({"layer": layer, "recoverability": sim})
        return pd.DataFrame(rows)

    def bootstrap_recoverability(
        self,
        deltas=None,
        k=1,
        n_boot=100,
        unit_col="sentence_id",
        layer_col="layer",
        delta_col="delta",
        seed=None,
    ):
        if deltas is None:
            deltas = self.deltas
        if seed is None:
            seed = self.seed
        boot_seeds = np.random.SeedSequence(seed).spawn(n_boot)

        all_results = []

        for i in range(n_boot):
            current_seed = int(boot_seeds[i].generate_state(1)[0])
            train, test = self.train_test_split(deltas, current_seed)
            curve = self.recoverability_curve(train, test, k = k)
            curve['boot_id'] = i
            all_results.append(curve)
        results = pd.concat(all_results, ignore_index=True)
        return results

    def compare_bootstrap_dfs(
        self,
        df_a,
        df_b,
        layer_col="layer",
        value_col="recoverability",
        boot_col="boot_id",
        n_perm=2000,
        n_boot_ci=2000,
        seed=None,
    ):
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(int(seed))

        merged = df_a.merge(df_b, on=[layer_col, boot_col], suffixes=("_a", "_b"))
        if merged.empty:
            return pd.DataFrame()

        rows = []
        for layer, g in merged.groupby(layer_col):
            vals = (g[f"{value_col}_a"] - g[f"{value_col}_b"]).to_numpy(dtype=np.float64)
            n = len(vals)
            if n < 2:
                continue

            obs_mean = float(np.mean(vals))

            # permutation p-value (paired sign-flip)
            perm_means = np.empty(int(n_perm), dtype=np.float64)
            for j in range(int(n_perm)):
                signs = rng.integers(0, 2, size=n) * 2 - 1
                perm_means[j] = float(np.mean(vals * signs))
            p_val = float((1 + np.sum(np.abs(perm_means) >= abs(obs_mean))) / (1 + len(perm_means)))

            # bootstrap CI on mean difference (optional but recommended)
            boot_means = np.empty(int(n_boot_ci), dtype=np.float64)
            for j in range(int(n_boot_ci)):
                idx = rng.integers(0, n, size=n)
                boot_means[j] = float(np.mean(vals[idx]))
            lo = float(np.quantile(boot_means, 0.025))
            hi = float(np.quantile(boot_means, 0.975))

            rows.append(
                {
                    layer_col: layer,
                    "mean_diff": obs_mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p_val,
                    "n": int(n),
                    "n_perm": int(n_perm),
                }
            )

        return pd.DataFrame(rows)


def demo_recoverability_curve():
    config = Config("config/standard.yaml")

    # Load our sentiment deltas without any recomputation
    deltas = Deltas(None)
    sentiment_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_deltas.csv")
    deltas.df = sentiment_deltas_df

    concept = Concept(None, PCA)
    sentiment_train, sentiment_test = concept.train_test_split(sentiment_deltas_df) 

    sentiment_null_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_null_deltas.csv")
    sentiment_null_train, sentiment_null_train = concept.train_test_split(sentiment_null_deltas_df)

    sentiment_recovery_df = concept.recoverability_curve(train_df=sentiment_train,
                                                     test_df=sentiment_test,
                                                     delta_col="delta",
                                                     k=1,
                                                     layer_col="layer") 

def main():
    config = Config("config/standard.yaml")
    deltas = Deltas(None)
    sentiment_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_deltas.csv")
    sentiment_null_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_null_deltas.csv")
    concept = Concept(None, PCA, config = config) 
    sentiment_boot = concept.bootstrap_recoverability(sentiment_deltas_df)
    null_boot = concept.bootstrap_recoverability(sentiment_null_deltas_df)
    result = concept.compare_bootstrap_dfs(sentiment_boot, null_boot)
    print(result)

    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS, config = config)
    df = temp.generate_all_permutation()
    gpt = GPT2()
    df = gpt.add_x_residuals_to_df(df)
    deltas = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = deltas.compute_mu(group_cols)
    deltas_adj = deltas.compute_adjacent_deltas(mu, group_cols)
    label_permutation_boot = concept.bootstrap_recoverability(deltas= deltas_adj)
    result_permutation = concept.compare_bootstrap_dfs(sentiment_boot, label_permutation_boot)
    print(result_permutation)
    
if __name__ == "__main__":
    main()
