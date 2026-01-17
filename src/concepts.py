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

    def construct_label_permutation_null(self):
        pass


def main():
    config = Config("config/standard.yaml")

    # Load our sentiment deltas without any recomputation
    deltas = Deltas(None)
    sentiment_deltas_df = deltas.load_deltas("cache/gpt2_sentiment_deltas.csv")
    deltas.df = sentiment_deltas_df

    concept = Concept(None, PCA)
    sentiment_train, sentiment_test = concept.train_test_split(sentiment_deltas_df) 

    temp = Template(SENTIMENT_SENTENCES, NULL_WORDS)
    gpt = GPT2()
    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df=df, x=None)
    delta = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj_null = delta.compute_adjacent_deltas(mu, group_cols)
    delta.cache_deltas("cache/gpt2_sentiment_null_deltas.csv")
    sentiment_null_train, sentiment_null_train = concept.train_test_split(deltas_adj_null)
    
if __name__ == "__main__":
    main()
