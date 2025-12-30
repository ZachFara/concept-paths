import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capture import GPT2
from src.templates import SENTIMENT_WORDS, SENTIMENT_SENTENCES, Template
from src.deltas import Deltas

class PCA:

    def __init__(self, adjacent_deltas_df):
        self.adjacent_deltas_df = adjacent_deltas_df

    def _compute_pca_for_one_layer(self, layer, df=None, n_components=None):
 
        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

        layer_df = deltas_adj[deltas_adj["layer"] == layer]
        if layer_df.empty:
            raise ValueError(f"No deltas for layer {layer}")

        X = np.stack([d.cpu().numpy() for d in layer_df["delta"].values])
        pca = SklearnPCA(n_components=n_components)
        pca.fit(X)

        return pca

    def get_all_layer_pca(self, df=None, n_components=None):

        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

        layers = sorted(deltas_adj["layer"].unique())
        pca_by_layer = {}
        for layer in layers:
            pca_by_layer[layer] = self._compute_pca_for_one_layer(
                layer, df=deltas_adj, n_components=n_components
            )
        return pca_by_layer

    def compute_pca_metrics(self, df=None, k_list=None, n_pc=10):
        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

        if k_list is None:
            k_list = [80, 85, 90, 95, 99]

        layers = sorted(deltas_adj["layer"].unique())
        rows = []
        for layer in layers:
            pca = self._compute_pca_for_one_layer(layer, df=deltas_adj, n_components=None)
            ratios = pca.explained_variance_ratio_
            cumulative = np.cumsum(ratios)
            row = {"layer": layer, "total_components": len(ratios)}

            for i in range(1, n_pc + 1):
                key = f"pc{i}"
                row[key] = float(ratios[i - 1]) if i - 1 < len(ratios) else None

            for k in k_list:
                threshold = float(k)
                if threshold > 1.0:
                    threshold = threshold / 100.0
                idx = int(np.searchsorted(cumulative, threshold, side="left"))
                row[f"k{int(k)}"] = idx + 1 if len(cumulative) else None

            rows.append(row)

        return pd.DataFrame(rows)

    def jackknife_pca_metrics(self, df=None, id_col="sentence_id", k_list=None, n_pc=10):
        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

        results = []
        for item_id in deltas_adj[id_col].unique():
            subset = deltas_adj[deltas_adj[id_col] != item_id]
            metrics = self.compute_pca_metrics(
                df=subset, k_list=k_list, n_pc=n_pc
            )
            if metrics.empty:
                continue
            metrics = metrics.copy()
            metrics["left_out"] = item_id
            results.append(metrics)

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    @staticmethod
    def _principal_angles(U, V):
        # U, V: [d, k] with orthonormal columns
        s = np.linalg.svd(U.T @ V, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        angles = np.arccos(s)
        return angles

    def compute_principal_angles(self, k=5, variance_threshold=None, df=None):
        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

        layers = sorted(deltas_adj["layer"].unique())
        pca_by_layer = self.get_all_layer_pca(df=deltas_adj, n_components=None)

        rows = []
        for i in range(len(layers) - 1):
            l1 = layers[i]
            l2 = layers[i + 1]
            pca1 = pca_by_layer[l1]
            pca2 = pca_by_layer[l2]

            if variance_threshold is not None:
                cum1 = np.cumsum(pca1.explained_variance_ratio_)
                cum2 = np.cumsum(pca2.explained_variance_ratio_)
                k1 = int(np.searchsorted(cum1, variance_threshold, side="left")) + 1
                k2 = int(np.searchsorted(cum2, variance_threshold, side="left")) + 1
                k_used = min(k1, k2)
            else:
                k_used = k

            U = pca1.components_[:k_used].T
            V = pca2.components_[:k_used].T
            angles = self._principal_angles(U, V)

            rows.append(
                {
                    "layer_from": l1,
                    "layer_to": l2,
                    "k_used": int(k_used),
                    "variance_threshold": variance_threshold,
                    "mean_angle": float(np.mean(angles)),
                    "max_angle": float(np.max(angles)),
                    "angles": angles,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _procrustes_rotation(U, V):
        # U, V: [d, k] with orthonormal columns
        M = U.T @ V
        Q, _, R_t = np.linalg.svd(M, full_matrices=False)
        R = Q @ R_t
        return R

    def compute_procrustes_alignment(self, k=5, variance_threshold=None, df=None):
        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

        layers = sorted(deltas_adj["layer"].unique())
        pca_by_layer = self.get_all_layer_pca(df=deltas_adj, n_components=None)

        rows = []
        for i in range(len(layers) - 1):
            l1 = layers[i]
            l2 = layers[i + 1]
            pca1 = pca_by_layer[l1]
            pca2 = pca_by_layer[l2]

            if variance_threshold is not None:
                cum1 = np.cumsum(pca1.explained_variance_ratio_)
                cum2 = np.cumsum(pca2.explained_variance_ratio_)
                k1 = int(np.searchsorted(cum1, variance_threshold, side="left")) + 1
                k2 = int(np.searchsorted(cum2, variance_threshold, side="left")) + 1
                k_used = min(k1, k2)
            else:
                k_used = k

            U = pca1.components_[:k_used].T
            V = pca2.components_[:k_used].T
            R = self._procrustes_rotation(U, V)
            U_rot = U @ R
            residual = float(np.linalg.norm(U_rot - V, ord="fro"))

            rows.append(
                {
                    "layer_from": l1,
                    "layer_to": l2,
                    "k_used": int(k_used),
                    "variance_threshold": variance_threshold,
                    "residual_fro": residual,
                    "rotation": R,
                }
            )
        return pd.DataFrame(rows)

def main():

    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()

    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df = df, x = None)

    delta = Deltas(df)

    group_cols = ["sentence_id", "layer"]

    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)

    pca = PCA(deltas_adj)

    metrics = pca.compute_pca_metrics(k_list=[80, 85, 90, 95, 99], n_pc=10)
    metrics.to_csv("outputs/pca_metrics.csv", index=False)

    jack = pca.jackknife_pca_metrics(k_list=[80, 85, 90, 95, 99], n_pc=10)
    jack.to_csv("outputs/pca_metrics_jackknife.csv", index=False)

    angles_k = pca.compute_principal_angles(k=5)
    angles_k.to_csv("outputs/pca_angles_k5.csv")

    angles_var = pca.compute_principal_angles(variance_threshold=0.9)
    angles_var.to_csv("outputs/pca_angles_var90.csv")

    proc_k = pca.compute_procrustes_alignment(k=5)
    proc_k.to_csv("outputs/pca_procrustes_k5.csv")

    proc_var = pca.compute_procrustes_alignment(variance_threshold=0.9)
    proc_var.to_csv("outputs/pca_procrustes_var90.csv")

if __name__ == "__main__":
    main()
