import numpy as np
import pandas as pd
from tqdm import tqdm

from src.templates import Template, NULL_WORDS, SENTIMENT_SENTENCES, SENTIMENT_WORDS
from src.capture import GPT2
from src.deltas import Deltas
from src.stats import Stats
from src.pca import PCA

class Bootstrap:
    def __init__(self, deltas_adj_df, id_col="sentence_id"):
        self.deltas_adj_df = deltas_adj_df
        self.id_col = id_col

    def _sample_ids(self, rng):
        ids = self.deltas_adj_df[self.id_col].unique()
        return rng.choice(ids, size=len(ids), replace=True)

    def _bootstrap_df(self, rng):
        sampled_ids = self._sample_ids(rng)
        parts = [self.deltas_adj_df[self.deltas_adj_df[self.id_col] == i] for i in sampled_ids]
        return pd.concat(parts, ignore_index=True)

    def bootstrap_step_consistency(self, n_boot=100, seed=0):
        rng = np.random.default_rng(seed)
        results = []
        for b in tqdm(range(n_boot), desc="bootstrap_step_consistency"):
            boot_df = self._bootstrap_df(rng)
            analysis = Stats(boot_df).step_consistency()
            analysis["boot_id"] = b
            results.append(analysis)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def bootstrap_axis_consistency(self, n_boot=100, seed=0):
        rng = np.random.default_rng(seed)
        results = []
        for b in tqdm(range(n_boot), desc="bootstrap_axis_consistency"):
            boot_df = self._bootstrap_df(rng)
            analysis = Stats(boot_df).axis_consistency()
            analysis["boot_id"] = b
            results.append(analysis)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def bootstrap_pca_metrics(self, n_boot=100, seed=0, k_list=None, n_pc=10):
        rng = np.random.default_rng(seed)
        results = []
        for b in tqdm(range(n_boot), desc="bootstrap_pca_metrics"):
            boot_df = self._bootstrap_df(rng)
            analysis = PCA(boot_df).compute_pca_metrics(
                k_list=k_list, n_pc=n_pc
            )
            analysis["boot_id"] = b
            results.append(analysis)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def bootstrap_principal_angles(self, n_boot=100, seed=0, k=5, variance_threshold=None):
        rng = np.random.default_rng(seed)
        results = []
        for b in tqdm(range(n_boot), desc="bootstrap_principal_angles"):
            boot_df = self._bootstrap_df(rng)
            analysis = PCA(boot_df).compute_principal_angles(
                k=k, variance_threshold=variance_threshold
            )
            analysis["boot_id"] = b
            results.append(analysis)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def bootstrap_procrustes_alignment(self, n_boot=100, seed=0, k=5, variance_threshold=None):
        rng = np.random.default_rng(seed)
        results = []
        for b in tqdm(range(n_boot), desc="bootstrap_procrustes_alignment"):
            boot_df = self._bootstrap_df(rng)
            analysis = PCA(boot_df).compute_procrustes_alignment(
                k=k, variance_threshold=variance_threshold
            )
            analysis["boot_id"] = b
            results.append(analysis)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def main():
    gpt = GPT2()

    sentiment_df = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS).get_all_sentences()
    sentiment_df = gpt.add_x_residuals_to_df(df=sentiment_df, x=None)
    sentiment_delta = Deltas(sentiment_df)
    sentiment_delta.df["level"] = sentiment_delta.df["level_id"].apply(
        sentiment_delta.level_to_int
    )
    sentiment_delta.ensure_hidden_last()
    sentiment_mu = sentiment_delta.compute_mu(group_cols=["sentence_id", "layer"])
    sentiment_deltas_adj = sentiment_delta.compute_adjacent_deltas(
        sentiment_mu, ["sentence_id", "layer"]
    )

    null_df = Template(SENTIMENT_SENTENCES, NULL_WORDS).get_all_sentences()
    null_df = gpt.add_x_residuals_to_df(df=null_df, x=None)
    null_delta = Deltas(null_df)
    null_delta.df["level"] = null_delta.df["level_id"].apply(null_delta.level_to_int)
    null_delta.ensure_hidden_last()
    null_mu = null_delta.compute_mu(group_cols=["sentence_id", "layer"])
    null_deltas_adj = null_delta.compute_adjacent_deltas(
        null_mu, ["sentence_id", "layer"]
    )

    sentiment_boot = Bootstrap(sentiment_deltas_adj)
    null_boot = Bootstrap(null_deltas_adj)

    sentiment_boot.bootstrap_step_consistency().to_csv(
        "outputs/step_consistency_bootstrap.csv", index=False
    )
    sentiment_boot.bootstrap_axis_consistency().to_csv(
        "outputs/axis_consistency_bootstrap.csv", index=False
    )
    sentiment_boot.bootstrap_pca_metrics().to_csv(
        "outputs/pca_metrics_bootstrap.csv", index=False
    )
    sentiment_boot.bootstrap_principal_angles(k=5).to_csv(
        "outputs/pca_angles_bootstrap_k5.csv", index=False
    )
    sentiment_boot.bootstrap_principal_angles(variance_threshold=0.9).to_csv(
        "outputs/pca_angles_bootstrap_var90.csv", index=False
    )
    sentiment_boot.bootstrap_procrustes_alignment(k=5).to_csv(
        "outputs/pca_procrustes_bootstrap_k5.csv", index=False
    )
    sentiment_boot.bootstrap_procrustes_alignment(variance_threshold=0.9).to_csv(
        "outputs/pca_procrustes_bootstrap_var90.csv", index=False
    )

    null_boot.bootstrap_step_consistency().to_csv(
        "outputs/null_step_consistency_bootstrap.csv", index=False
    )
    null_boot.bootstrap_axis_consistency().to_csv(
        "outputs/null_axis_consistency_bootstrap.csv", index=False
    )
    null_boot.bootstrap_pca_metrics().to_csv(
        "outputs/null_pca_metrics_bootstrap.csv", index=False
    )
    null_boot.bootstrap_principal_angles(k=5).to_csv(
        "outputs/null_pca_angles_bootstrap_k5.csv", index=False
    )
    null_boot.bootstrap_principal_angles(variance_threshold=0.9).to_csv(
        "outputs/null_pca_angles_bootstrap_var90.csv", index=False
    )
    null_boot.bootstrap_procrustes_alignment(k=5).to_csv(
        "outputs/null_pca_procrustes_bootstrap_k5.csv", index=False
    )
    null_boot.bootstrap_procrustes_alignment(variance_threshold=0.9).to_csv(
        "outputs/null_pca_procrustes_bootstrap_var90.csv", index=False
    )

if __name__ == "__main__":
    main()
