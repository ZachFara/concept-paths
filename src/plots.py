
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class Plots:
    def __init__(self):
        pass

    def _load_df(self, df=None, csv_path=None):
        if df is not None:
            return df
        if csv_path is None:
            raise ValueError("Provide df or csv_path")
        return pd.read_csv(csv_path)

    def plot_axis_consistency(self, df=None, csv_path=None, output_path=None, title_suffix=None):
        # Jackknifed axis consistency: mean and sem across left_out per layer.
        data = self._load_df(df=df, csv_path=csv_path)
        grouped = data.groupby("layer")["G"]
        summary = grouped.agg(["mean", "sem", "count"]).reset_index()
        summary["t_crit"] = summary["count"].apply(
            lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
        )
        summary["ci_half"] = summary["t_crit"] * summary["sem"]

        fig, ax = plt.subplots()
        ax.plot(summary["layer"], summary["mean"], label="axis consistency")
        ax.fill_between(
            summary["layer"],
            summary["mean"] - summary["ci_half"],
            summary["mean"] + summary["ci_half"],
            alpha=0.2,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        title = "Axis Consistency"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend()

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

    def plot_step_consistency(self, df=None, csv_path=None, output_path=None, title_suffix=None):
        # Jackknifed step consistency: mean and sem across left_out per step and layer.
        data = self._load_df(df=df, csv_path=csv_path)
        grouped = data.groupby(["level_from", "level_to", "layer"])["G"]
        summary = grouped.agg(["mean", "sem", "count"]).reset_index()
        summary["t_crit"] = summary["count"].apply(
            lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
        )
        summary["ci_half"] = summary["t_crit"] * summary["sem"]

        fig, ax = plt.subplots()
        for (level_from, level_to), g in summary.groupby(["level_from", "level_to"]):
            g = g.sort_values("layer")
            label = f"{level_from}->{level_to}"
            ax.plot(g["layer"], g["mean"], label=label)
            ax.fill_between(
                g["layer"],
                g["mean"] - g["ci_half"],
                g["mean"] + g["ci_half"],
                alpha=0.2,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        title = "Step Consistency"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend(ncol=2, fontsize=8)

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

    def plot_pca_components(
        self,
        df=None,
        csv_path=None,
        components=None,
        output_path=None,
        cumulative=False,
        title_suffix=None,
    ):
        # Jackknifed PCA component variance: mean and CI across left_out per layer.
        data = self._load_df(df=df, csv_path=csv_path)
        if components is None:
            components = [1, 2, 3, 4, 5]

        fig, ax = plt.subplots()
        for comp in components:
            cols = [f"pc{i}" for i in range(1, int(comp) + 1)]
            if not all(c in data.columns for c in cols):
                continue
            if cumulative:
                data = data.copy()
                data["_pc_sum"] = data[cols].sum(axis=1)
                grouped = data.groupby("layer")["_pc_sum"]
            else:
                col = f"pc{int(comp)}"
                grouped = data.groupby("layer")[col]
            summary = grouped.agg(["mean", "sem", "count"]).reset_index()
            summary["t_crit"] = summary["count"].apply(
                lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
            )
            summary["ci_half"] = summary["t_crit"] * summary["sem"]
            label = f"pc{int(comp)}"
            if cumulative:
                label = f"pc1-{int(comp)}"
            ax.plot(summary["layer"], summary["mean"], label=label)
            ax.fill_between(
                summary["layer"],
                summary["mean"] - summary["ci_half"],
                summary["mean"] + summary["ci_half"],
                alpha=0.2,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Explained variance ratio")
        title = "PCA Components"
        if cumulative:
            title = "PCA Cumulative Components"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend(ncol=2, fontsize=8)

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

    def plot_pca_k_thresholds(self, df=None, csv_path=None, thresholds=None, output_path=None, title_suffix=None):
        # Jackknifed k-thresholds: mean and CI across left_out per layer.
        data = self._load_df(df=df, csv_path=csv_path)
        if thresholds is None:
            thresholds = [80, 85, 90, 95, 99]

        fig, ax = plt.subplots()
        for k in thresholds:
            col = f"k{int(k)}"
            if col not in data.columns:
                continue
            grouped = data.groupby("layer")[col]
            summary = grouped.agg(["mean", "sem", "count"]).reset_index()
            summary["t_crit"] = summary["count"].apply(
                lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
            )
            summary["ci_half"] = summary["t_crit"] * summary["sem"]
            ax.plot(summary["layer"], summary["mean"], label=col)
            ax.fill_between(
                summary["layer"],
                summary["mean"] - summary["ci_half"],
                summary["mean"] + summary["ci_half"],
                alpha=0.2,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Components to reach threshold")
        title = "PCA Thresholds"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend(ncol=2, fontsize=8)

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

    def _k_used_summary(self, data, layer_col):
        if "k_used" not in data.columns:
            return None
        if "variance_threshold" in data.columns:
            data = data[data["variance_threshold"].notna()]
        if data.empty:
            return None
        k_used = data.groupby(layer_col)["k_used"].agg(
            lambda s: int(s.mode().iloc[0]) if not s.mode().empty else int(s.iloc[0])
        )
        return k_used.reset_index()

    def _annotate_k_used(self, ax, summary, x_col, y_col):
        if "k_used" not in summary.columns:
            return
        for _, row in summary.iterrows():
            ax.annotate(
                f"k={int(row['k_used'])}",
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                alpha=0.8,
            )

    def plot_principal_angles(self, df=None, csv_path=None, output_path=None, title_suffix=None):
        data = self._load_df(df=df, csv_path=csv_path)
        grouped = data.groupby("layer_from")["mean_angle"]
        summary = grouped.agg(["mean", "sem", "count"]).reset_index()
        summary["t_crit"] = summary["count"].apply(
            lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
        )
        summary["ci_half"] = summary["t_crit"] * summary["sem"]

        fig, ax = plt.subplots()
        ax.plot(summary["layer_from"], summary["mean"], label="principal angle")
        ax.fill_between(
            summary["layer_from"],
            summary["mean"] - summary["ci_half"],
            summary["mean"] + summary["ci_half"],
            alpha=0.2,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Angle (radians)")
        title = "Principal Angles (Adjacent Layers)"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend()

        k_summary = self._k_used_summary(data, "layer_from")
        if k_summary is not None:
            summary_k = summary.merge(k_summary, on="layer_from", how="left")
            self._annotate_k_used(ax, summary_k, "layer_from", "mean")

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

    def plot_procrustes_alignment(self, df=None, csv_path=None, output_path=None, title_suffix=None):
        data = self._load_df(df=df, csv_path=csv_path)
        grouped = data.groupby("layer_from")["residual_fro"]
        summary = grouped.agg(["mean", "sem", "count"]).reset_index()
        summary["t_crit"] = summary["count"].apply(
            lambda n: stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
        )
        summary["ci_half"] = summary["t_crit"] * summary["sem"]

        fig, ax = plt.subplots()
        ax.plot(summary["layer_from"], summary["mean"], label="procrustes residual")
        ax.fill_between(
            summary["layer_from"],
            summary["mean"] - summary["ci_half"],
            summary["mean"] + summary["ci_half"],
            alpha=0.2,
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel("Frobenius residual")
        title = "Procrustes Alignment (Adjacent Layers)"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend()

        k_summary = self._k_used_summary(data, "layer_from")
        if k_summary is not None:
            summary_k = summary.merge(k_summary, on="layer_from", how="left")
            self._annotate_k_used(ax, summary_k, "layer_from", "mean")

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

def main():
    plots = Plots()
    plots.plot_axis_consistency(
        csv_path="outputs/axis_consistency.csv",
        output_path="outputs/axis_consistency.png",
        title_suffix="Jackknife",
    )
    plots.plot_step_consistency(
        csv_path="outputs/step_consistency.csv",
        output_path="outputs/step_consistency.png",
        title_suffix="Jackknife",
    )
    plots.plot_pca_components(
        components = [1, 2, 3],
        cumulative = True,
        csv_path="outputs/pca_metrics_jackknife.csv",
        output_path="outputs/pca_components.png",
        title_suffix="Jackknife",
    )
    plots.plot_pca_k_thresholds(
        thresholds= [85, 90, 95],
        csv_path="outputs/pca_metrics_jackknife.csv",
        output_path="outputs/pca_thresholds.png",
        title_suffix="Jackknife",
    )
    plots.plot_principal_angles(
        csv_path="outputs/pca_angles_k5.csv",
        output_path="outputs/pca_angles_k5.png",
        title_suffix="Jackknife",
    )
    plots.plot_principal_angles(
        csv_path="outputs/pca_angles_var90.csv",
        output_path="outputs/pca_angles_var90.png",
        title_suffix="Jackknife",
    )
    plots.plot_procrustes_alignment(
        csv_path="outputs/pca_procrustes_k5.csv",
        output_path="outputs/pca_procrustes_k5.png",
        title_suffix="Jackknife",
    )
    plots.plot_procrustes_alignment(
        csv_path="outputs/pca_procrustes_var90.csv",
        output_path="outputs/pca_procrustes_var90.png",
        title_suffix="Jackknife",
    )

    plots.plot_axis_consistency(
        csv_path="outputs/axis_consistency_bootstrap.csv",
        output_path="outputs/axis_consistency_bootstrap.png",
        title_suffix="Bootstrap",
    )
    plots.plot_step_consistency(
        csv_path="outputs/step_consistency_bootstrap.csv",
        output_path="outputs/step_consistency_bootstrap.png",
        title_suffix="Bootstrap",
    )
    plots.plot_pca_components(
        components=[1, 2, 3],
        cumulative=True,
        csv_path="outputs/pca_metrics_bootstrap.csv",
        output_path="outputs/pca_components_bootstrap.png",
        title_suffix="Bootstrap",
    )
    plots.plot_pca_k_thresholds(
        thresholds=[85, 90, 95],
        csv_path="outputs/pca_metrics_bootstrap.csv",
        output_path="outputs/pca_thresholds_bootstrap.png",
        title_suffix="Bootstrap",
    )
    plots.plot_principal_angles(
        csv_path="outputs/pca_angles_bootstrap_k5.csv",
        output_path="outputs/pca_angles_bootstrap_k5.png",
        title_suffix="Bootstrap",
    )
    plots.plot_principal_angles(
        csv_path="outputs/pca_angles_bootstrap_var90.csv",
        output_path="outputs/pca_angles_bootstrap_var90.png",
        title_suffix="Bootstrap",
    )
    plots.plot_procrustes_alignment(
        csv_path="outputs/pca_procrustes_bootstrap_k5.csv",
        output_path="outputs/pca_procrustes_bootstrap_k5.png",
        title_suffix="Bootstrap",
    )
    plots.plot_procrustes_alignment(
        csv_path="outputs/pca_procrustes_bootstrap_var90.csv",
        output_path="outputs/pca_procrustes_bootstrap_var90.png",
        title_suffix="Bootstrap",
    )

if __name__ == "__main__":
    main()
