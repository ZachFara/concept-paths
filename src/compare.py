import os
import numpy as np
import pandas as pd

from src.config import Config

K_LIST = [85, 90, 95]


class Comparison:
    def __init__(
        self,
        alt_df=None,
        alt_csv_path=None,
        null_df=None,
        null_csv_path=None,
        seed = 0,
        config:Config = None
    ):
        self.alt_df = self._load_df(alt_df, alt_csv_path)
        self.null_df = self._load_df(null_df, null_csv_path)

        if config:
            self.seed = config.random_seed
        else:
            self.seed = seed

        assert self.seed is not None

    def _load_df(self, df=None, csv_path=None):
        if df is not None:
            return df
        if csv_path is None:
            raise ValueError("Provide df or csv_path")
        return pd.read_csv(csv_path)

    def bootstrap_diff(self, alt_vals, null_vals, n_boot, rng):
        if len(alt_vals) == 0 or len(null_vals) == 0:
            raise ValueError("Alternate or null values are empty")

        diffs = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            a = rng.choice(alt_vals, size=len(alt_vals), replace=True)
            n = rng.choice(null_vals, size=len(null_vals), replace=True)
            diffs[i] = np.mean(a) - np.mean(n)
        return diffs

    def compare_bootstrap(
        self,
        metric_cols,
        group_cols,
        n_boot=1000,
        seed=None,
        fdr=True,
    ):

        if seed is None:
            seed = self.seed

        rng = np.random.default_rng(seed)
        rows = []
        for metric in metric_cols:
            for group_key, g_alt in self.alt_df.groupby(group_cols):
                if group_cols:
                    if not isinstance(group_key, tuple):
                        group_key = (group_key,)
                    mask = (
                        self.null_df[group_cols]
                        == pd.Series(group_key, index=group_cols)
                    ).all(axis=1)
                    g_null = self.null_df[mask]
                else:
                    g_null = self.null_df

                alt_vals = g_alt[metric].dropna().to_numpy()
                null_vals = g_null[metric].dropna().to_numpy()
                if len(alt_vals) == 0 or len(null_vals) == 0:
                    continue

                mean_alt = float(np.mean(alt_vals))
                mean_null = float(np.mean(null_vals))
                diff = mean_alt - mean_null

                diffs = self.bootstrap_diff(alt_vals, null_vals, n_boot, rng)
                ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
                p_low = np.mean(diffs <= 0)
                p_high = np.mean(diffs >= 0)
                p_two_sided = float(2 * min(p_low, p_high))

                a_std = float(np.std(alt_vals, ddof=1)) if len(alt_vals) > 1 else np.nan
                n_std = float(np.std(null_vals, ddof=1)) if len(null_vals) > 1 else np.nan
                pooled = (
                    np.sqrt((a_std ** 2 + n_std ** 2) / 2)
                    if np.isfinite(a_std) and np.isfinite(n_std)
                    else np.nan
                )
                cohens_d = diff / pooled if pooled and np.isfinite(pooled) else np.nan

                row = {
                    "metric": metric,
                    "mean_alt": mean_alt,
                    "mean_null": mean_null,
                    "diff_mean": diff,
                    "diff_ci_low": float(ci_low),
                    "diff_ci_high": float(ci_high),
                    "p_two_sided": p_two_sided,
                    "cohens_d": cohens_d,
                    "n_alt": len(alt_vals),
                    "n_null": len(null_vals),
                }
                if group_cols:
                    for col, val in zip(group_cols, group_key):
                        row[col] = val
                rows.append(row)

        out_df = pd.DataFrame(rows)
        if fdr and not out_df.empty and "metric" in out_df.columns:
            out_df["q_value"] = np.nan
            for metric, g in out_df.groupby("metric"):
                qvals = _fdr_bh(g["p_two_sided"].to_numpy())
                out_df.loc[g.index, "q_value"] = qvals
        return out_df


def _pc_cols(n=10):
    return [f"pc{i}" for i in range(1, n + 1)]


def _k_cols(k_list):
    return [f"k{int(k)}" for k in k_list]


def _fdr_bh(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    qvals = np.empty(n, dtype=np.float64)
    qvals[order] = pvals[order] * n / ranks
    qvals = np.minimum.accumulate(qvals[order][::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(qvals, 1.0)
    return out


def main():
    os.makedirs("outputs", exist_ok=True)
    config = Config("config/test.yaml")
    seed = config.get('random_seed', 0)
    n_boot = config.get('n_boot', 1000)

    configs = [
        {
            "name": "step_consistency",
            "alt_path": "outputs/gpt/sentiment/data/bootstrap/step_consistency_bootstrap.csv",
            "null_path": "outputs/gpt/sentiment/data/bootstrap/null_step_consistency_bootstrap.csv",
            "group_cols": ["layer", "level_from", "level_to"],
            "metric_cols": ["G"],
        },
        {
            "name": "axis_consistency",
            "alt_path": "outputs/gpt/sentiment/data/bootstrap/axis_consistency_bootstrap.csv",
            "null_path": "outputs/gpt/sentiment/data/bootstrap/null_axis_consistency_bootstrap.csv",
            "group_cols": ["layer"],
            "metric_cols": ["G"],
        },
        {
            "name": "pca_metrics",
            "alt_path": "outputs/gpt/sentiment/data/bootstrap/pca_metrics_bootstrap.csv",
            "null_path": "outputs/gpt/sentiment/data/bootstrap/null_pca_metrics_bootstrap.csv",
            "group_cols": ["layer"],
            "metric_cols": _pc_cols(10) + _k_cols(K_LIST),
        },
        {
            "name": "pca_angles_k5", "alt_path": "outputs/gpt/sentiment/data/bootstrap/pca_angles_bootstrap_k5.csv", "null_path": "outputs/gpt/sentiment/data/bootstrap/null_pca_angles_bootstrap_k5.csv", "group_cols": ["layer_from", "layer_to"], "metric_cols": ["mean_angle", "max_angle"],
        },
        {
            "name": "pca_angles_var90",
            "alt_path": "outputs/gpt/sentiment/data/bootstrap/pca_angles_bootstrap_var90.csv",
            "null_path": "outputs/gpt/sentiment/data/bootstrap/null_pca_angles_bootstrap_var90.csv",
            "group_cols": ["layer_from", "layer_to"],
            "metric_cols": ["mean_angle", "max_angle", "k_used"],
        },
        {
            "name": "pca_procrustes_k5",
            "alt_path": "outputs/gpt/sentiment/data/bootstrap/pca_procrustes_bootstrap_k5.csv",
            "null_path": "outputs/gpt/sentiment/data/bootstrap/null_pca_procrustes_bootstrap_k5.csv",
            "group_cols": ["layer_from", "layer_to"],
            "metric_cols": ["residual_fro"],
        },
        {
            "name": "pca_procrustes_var90",
            "alt_path": "outputs/gpt/sentiment/data/bootstrap/pca_procrustes_bootstrap_var90.csv",
            "null_path": "outputs/gpt/sentiment/data/bootstrap/null_pca_procrustes_bootstrap_var90.csv",
            "group_cols": ["layer_from", "layer_to"],
            "metric_cols": ["residual_fro", "k_used"],
        },
    ]

    for cfg in configs:
        comp = Comparison(
            alt_csv_path=cfg["alt_path"],
            null_csv_path=cfg["null_path"],
        )
        out_df = comp.compare_bootstrap(
            metric_cols=cfg["metric_cols"],
            group_cols=cfg["group_cols"],
            n_boot=n_boot,
            seed=seed,
        )
        output_path = f"outputs/test/compare_{cfg['name']}.csv"
        out_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        if out_df.empty:
            continue
        if "cohens_d" in out_df.columns:
            top = out_df.reindex(out_df["cohens_d"].abs().sort_values(ascending=False).index)
            print(top.head(3).to_string(index=False))
        else:
            print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
