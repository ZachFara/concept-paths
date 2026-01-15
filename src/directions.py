"""
Directions: extract and compare layerwise direction trajectories from deltas.

Terminology
- Direction: the first principal component (pc1) per layer, normalized to unit length.
  The per-layer pc1 vectors form a "trajectory" across layers.
- Subspace: the top-k PCA components per layer (orthonormal basis).
- Spectra: explained_variance_ratio_ per layer (used for effective rank).
"""

from __future__ import annotations

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



class Directions:

    def __init__(self, deltas = None, pca_dict = None, top_k = 5, config = None, seed = 0):
        self.deltas = deltas
        self.pca_dict = pca_dict
        self.top_k = top_k

        if config:
            self.seed = config.random_seed
        else:
            self.seed = seed

    def resolve_param(self, param, default):
        return default if param is None else param

    @staticmethod
    def _unit(v, eps=1e-8):
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + eps)

    def build(self, deltas=None, pca_dict=None, top_k=None, sign_fix="prev"):
        # Resolve parameters
        deltas = self.resolve_param(deltas, self.deltas)
        pca_dict = self.resolve_param(pca_dict, self.pca_dict)
        top_k = self.resolve_param(top_k, self.top_k)

        layers = sorted(pca_dict.keys())
        logger.debug(f"Running Directions.build for layers: {layers}")

        rows = []
        prev_dir = None

        for layer in layers:
            pca = pca_dict[layer]

            # --- Direction (PC1) ---
            d = pca.components_[0]
            d = self._unit(d)

            if sign_fix == "prev" and prev_dir is not None:
                if np.dot(d, prev_dir) < 0:
                    d = -d
            prev_dir = d

            row = {
                "layer": layer,
                "direction": d.astype(np.float32),
            }

            # --- Subspace (top-k PCs) ---
            if top_k is not None and top_k > 1:
                k = min(top_k, pca.components_.shape[0])
                basis = pca.components_[:k]
                basis = basis / (np.linalg.norm(basis, axis=1, keepdims=True) + 1e-8)
                row["subspace"] = basis.astype(np.float32)

            # --- Spectrum ---
            if hasattr(pca, "explained_variance_ratio_"):
                row["spectrum"] = (
                    pca.explained_variance_ratio_[:top_k].astype(np.float32)
                    if top_k is not None
                    else pca.explained_variance_ratio_.astype(np.float32)
                )

            rows.append(row)

        df = pd.DataFrame(rows)

        # Explicitly sort again for safety
        df = df.sort_values("layer").reset_index(drop=True)

        return df

    def build_from_deltas(self, deltas, top_k=None, sign_fix="prev"):
        top_k = self.resolve_param(top_k, self.top_k)
        n_components = top_k if top_k and top_k > 0 else None
        pca = PCA(deltas)
        pca_dict = pca.get_all_layer_pca(df=deltas, n_components=n_components)
        return self.build(deltas=deltas, pca_dict=pca_dict, top_k=top_k, sign_fix=sign_fix)

    def trajectory_distance(self, df_a, df_b, reducer="mean"):

        # Align on layers explicitly
        merged = df_a.merge(df_b, on="layer", suffixes=("_a", "_b"))

        dots = np.array([
            abs(np.dot(a, b))
            for a, b in zip(merged["direction_a"], merged["direction_b"])
        ])

        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)

        if reducer == "max":
            return float(np.max(angles)), angles

        return float(np.mean(angles)), angles

    def bootstrap_distances(
        self,
        deltas,
        unit_col="sentence_id",
        n_boot=100,
        seed=self.seed,
        top_k=None,
        sign_fix="prev",
    ):
        rng = np.random.default_rng(int(seed))
        unit_ids = np.array(sorted(deltas[unit_col].unique()))
        n_units = len(unit_ids)
        if n_units == 0:
            return pd.DataFrame(columns=["distance"])
        rows = []
        for _ in tqdm(range(int(n_boot)), desc="bootstrap", leave=False):
            a_ids = rng.choice(unit_ids, size=n_units, replace=True)
            b_ids = rng.choice(unit_ids, size=n_units, replace=True)
            a_ids = set(a_ids.tolist())
            b_ids = set(b_ids.tolist())
            df_a = deltas[deltas[unit_col].isin(a_ids)]
            df_b = deltas[deltas[unit_col].isin(b_ids)]
            dir_a = self.build_from_deltas(df_a, top_k=top_k, sign_fix=sign_fix)
            dir_b = self.build_from_deltas(df_b, top_k=top_k, sign_fix=sign_fix)
            dist, _ = self.trajectory_distance(dir_a, dir_b)
            rows.append({"distance": dist})
        return pd.DataFrame(rows)

    def permutation_test_across(
        self,
        deltas_a,
        deltas_b,
        unit_col="sentence_id",
        n_perm=1000,
        seed=self.seed,
        top_k=None,
        sign_fix="prev",
    ):
        def with_unit_key(df, prefix):
            df = df.copy()
            df["_unit_key"] = df[unit_col].astype(str).map(lambda v: f"{prefix}:{v}")
            return df

        a = with_unit_key(deltas_a, "A")
        b = with_unit_key(deltas_b, "B")
        combined = pd.concat([a, b], ignore_index=True)

        unit_keys = combined["_unit_key"].unique()
        labels = np.array([1] * len(a["_unit_key"].unique()) + [0] * len(b["_unit_key"].unique()))
        if len(labels) != len(unit_keys):
            labels = np.array([1 if k.startswith("A:") else 0 for k in unit_keys])

        rng = np.random.default_rng(int(seed))
        observed_a = self.build_from_deltas(a, top_k=top_k, sign_fix=sign_fix)
        observed_b = self.build_from_deltas(b, top_k=top_k, sign_fix=sign_fix)
        observed, _ = self.trajectory_distance(observed_a, observed_b)

        stats = []
        for _ in tqdm(range(int(n_perm)), desc="permute", leave=False):
            rng.shuffle(labels)
            label_map = dict(zip(unit_keys, labels))
            combined["_perm_label"] = combined["_unit_key"].map(label_map)
            perm_a = combined[combined["_perm_label"] == 1].drop(columns=["_perm_label"])
            perm_b = combined[combined["_perm_label"] == 0].drop(columns=["_perm_label"])
            dir_a = self.build_from_deltas(perm_a, top_k=top_k, sign_fix=sign_fix)
            dir_b = self.build_from_deltas(perm_b, top_k=top_k, sign_fix=sign_fix)
            dist, _ = self.trajectory_distance(dir_a, dir_b)
            stats.append(dist)

        stats = np.array(stats, dtype=np.float32)
        p_value = float(np.mean(stats >= observed))
        return pd.DataFrame(
            [{"observed": observed, "p_value": p_value, "n_perm": int(n_perm)}]
        )

def main():
    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()
    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df=df, x=None)
    delta = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj_sentiment = delta.compute_adjacent_deltas(mu, group_cols)
    pca = PCA(deltas_adj_sentiment)
    pca_dict_sentiment = pca.get_all_layer_pca(df = deltas_adj_sentiment, n_components = 5)
    
    temp = Template(SENTIMENT_SENTENCES, NULL_WORDS)
    gpt = GPT2()
    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df=df, x=None)
    delta = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj_null = delta.compute_adjacent_deltas(mu, group_cols)
    pca = PCA(deltas_adj_null)
    pca_dict_null = pca.get_all_layer_pca(df = deltas_adj_null, n_components = 5)

    config = Config("config/standard.yaml")

    direction = Directions(config = config)
    directions_df_sentiment = direction.build(deltas_adj_sentiment, pca_dict_sentiment)
    directions_df_null = direction.build(deltas_adj_null, pca_dict_null)
    output = direction.trajectory_distance(directions_df_sentiment, directions_df_null)
    print(output)

    within_sent = direction.bootstrap_distances(
        deltas_adj_sentiment, unit_col="sentence_id", n_boot=200
    )
    within_null = direction.bootstrap_distances(
        deltas_adj_null, unit_col="sentence_id", n_boot=200
    )
    perm = direction.permutation_test_across(
        deltas_adj_sentiment, deltas_adj_null, unit_col="sentence_id", n_perm=500, seed=0
    )
    print(within_sent.describe())
    print(within_null.describe())
    print(perm)

if __name__ == "__main__":
    main()
