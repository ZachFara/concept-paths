import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.bootstrap import Bootstrap
from src.capture import GPT2
from src.templates import Template, SENTIMENT_WORDS, SENTIMENT_SENTENCES, NULL_WORDS
from src.compare import Comparison
from src.deltas import Deltas
from src.pca import PCA
from src.plots import Plots
from src.stats import Stats


class Pipeline:
    def __init__(
        self,
        output_path="outputs",
        variance_threshold=0.9,
        k_list=None,
        n_pc=10,
        n_boot_bootstrap=100,
        n_boot_compare=1000,
        k_rotation=5,
        stats_subdir="data/stats",
        pca_subdir="data/pca",
        bootstrap_subdir="data/bootstrap",
        compare_subdir="data/comparison",
        plots_subdir="plots",
    ):
        self.output_path = output_path
        self.variance_threshold = variance_threshold
        self.k_list = k_list if k_list is not None else [80, 85, 90, 95, 99]
        self.n_pc = n_pc
        self.n_boot_bootstrap = n_boot_bootstrap
        self.n_boot_compare = n_boot_compare
        self.k_rotation = k_rotation
        os.makedirs(self.output_path, exist_ok=True)
        self.stats_dir = self._out_dir(stats_subdir)
        self.pca_dir = self._out_dir(pca_subdir)
        self.bootstrap_dir = self._out_dir(bootstrap_subdir)
        self.compare_dir = self._out_dir(compare_subdir)
        self.plots_dir = self._out_dir(plots_subdir)

    def _out_dir(self, subdir):
        if not subdir:
            return self.output_path
        out_dir = os.path.join(self.output_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def get_deltas(self, template, model):
        df = template.get_all_sentences()
        df = model.add_x_residuals_to_df(df=df, x=None)
        delta = Deltas(df)
        delta.df["level"] = delta.df["level_id"].apply(delta.level_to_int)
        delta.ensure_hidden_last()
        mu = delta.compute_mu(group_cols=["sentence_id", "layer"])
        deltas_adj = delta.compute_adjacent_deltas(mu, ["sentence_id", "layer"])
        return deltas_adj

    def get_stats(self, deltas_adj, subdir=None):

        print("Running stats...")

        out_dir = self.stats_dir if subdir is None else self._out_dir(subdir)
        Stats(deltas_adj).axis_consistency().to_csv(
            os.path.join(out_dir, "axis_consistency.csv"), index=False
        )
        Stats(deltas_adj).step_consistency().to_csv(
            os.path.join(out_dir, "step_consistency.csv"), index=False
        )

    def run_pca(self, deltas_adj, subdir=None):

        print("Running PCA...")

        out_dir = self.pca_dir if subdir is None else self._out_dir(subdir)
        pca = PCA(deltas_adj)
        pca.compute_pca_metrics(k_list=self.k_list, n_pc=self.n_pc).to_csv(
            os.path.join(out_dir, "pca_metrics.csv"), index=False
        )
        pca.jackknife_pca_metrics(k_list=self.k_list, n_pc=self.n_pc).to_csv(
            os.path.join(out_dir, "pca_metrics_jackknife.csv"), index=False
        )
        pca.compute_principal_angles(k=self.k_rotation).to_csv(
            os.path.join(out_dir, f"pca_angles_k{self.k_rotation}.csv"), index=False
        )
        pca.compute_principal_angles(variance_threshold=self.variance_threshold).to_csv(
            os.path.join(out_dir, f"pca_angles_var{int(self.variance_threshold*100)}.csv"),
            index=False,
        )
        pca.compute_procrustes_alignment(k=self.k_rotation).to_csv(
            os.path.join(out_dir, f"pca_procrustes_k{self.k_rotation}.csv"), index=False
        )
        pca.compute_procrustes_alignment(variance_threshold=self.variance_threshold).to_csv(
            os.path.join(out_dir, f"pca_procrustes_var{int(self.variance_threshold*100)}.csv"),
            index=False,
        )

    def run_bootstrap(self, sentiment_deltas, null_deltas, subdir=None):

        print("Running bootstrap...")

        out_dir = self.bootstrap_dir if subdir is None else self._out_dir(subdir)
        sentiment_boot = Bootstrap(sentiment_deltas)
        null_boot = Bootstrap(null_deltas)

        sentiment_boot.bootstrap_step_consistency(n_boot=self.n_boot_bootstrap).to_csv(
            os.path.join(out_dir, "step_consistency_bootstrap.csv"), index=False
        )
        sentiment_boot.bootstrap_axis_consistency(n_boot=self.n_boot_bootstrap).to_csv(
            os.path.join(out_dir, "axis_consistency_bootstrap.csv"), index=False
        )
        sentiment_boot.bootstrap_pca_metrics(
            n_boot=self.n_boot_bootstrap, k_list=self.k_list, n_pc=self.n_pc
        ).to_csv(
            os.path.join(out_dir, "pca_metrics_bootstrap.csv"), index=False
        )
        sentiment_boot.bootstrap_principal_angles(
            n_boot=self.n_boot_bootstrap, k=self.k_rotation
        ).to_csv(
            os.path.join(out_dir, f"pca_angles_bootstrap_k{self.k_rotation}.csv"),
            index=False,
        )
        sentiment_boot.bootstrap_principal_angles(
            n_boot=self.n_boot_bootstrap, variance_threshold=self.variance_threshold
        ).to_csv(
            os.path.join(out_dir, f"pca_angles_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            index=False,
        )
        sentiment_boot.bootstrap_procrustes_alignment(
            n_boot=self.n_boot_bootstrap, k=self.k_rotation
        ).to_csv(
            os.path.join(out_dir, f"pca_procrustes_bootstrap_k{self.k_rotation}.csv"),
            index=False,
        )
        sentiment_boot.bootstrap_procrustes_alignment(
            n_boot=self.n_boot_bootstrap, variance_threshold=self.variance_threshold
        ).to_csv(
            os.path.join(out_dir, f"pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            index=False,
        )

        null_boot.bootstrap_step_consistency(n_boot=self.n_boot_bootstrap).to_csv(
            os.path.join(out_dir, "null_step_consistency_bootstrap.csv"), index=False
        )
        null_boot.bootstrap_axis_consistency(n_boot=self.n_boot_bootstrap).to_csv(
            os.path.join(out_dir, "null_axis_consistency_bootstrap.csv"), index=False
        )
        null_boot.bootstrap_pca_metrics(
            n_boot=self.n_boot_bootstrap, k_list=self.k_list, n_pc=self.n_pc
        ).to_csv(
            os.path.join(out_dir, "null_pca_metrics_bootstrap.csv"), index=False
        )
        null_boot.bootstrap_principal_angles(
            n_boot=self.n_boot_bootstrap, k=self.k_rotation
        ).to_csv(
            os.path.join(out_dir, f"null_pca_angles_bootstrap_k{self.k_rotation}.csv"),
            index=False,
        )
        null_boot.bootstrap_principal_angles(
            n_boot=self.n_boot_bootstrap, variance_threshold=self.variance_threshold
        ).to_csv(
            os.path.join(out_dir, f"null_pca_angles_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            index=False,
        )
        null_boot.bootstrap_procrustes_alignment(
            n_boot=self.n_boot_bootstrap, k=self.k_rotation
        ).to_csv(
            os.path.join(out_dir, f"null_pca_procrustes_bootstrap_k{self.k_rotation}.csv"),
            index=False,
        )
        null_boot.bootstrap_procrustes_alignment(
            n_boot=self.n_boot_bootstrap, variance_threshold=self.variance_threshold
        ).to_csv(
            os.path.join(out_dir, f"null_pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            index=False,
        )

    def run_comparison(self, subdir=None):

        print("Running comparison...")

        out_dir = self.compare_dir if subdir is None else self._out_dir(subdir)
        in_dir = self.bootstrap_dir
        Comparison(
            alt_csv_path=os.path.join(in_dir, "step_consistency_bootstrap.csv"),
            null_csv_path=os.path.join(in_dir, "null_step_consistency_bootstrap.csv"),
        ).compare_bootstrap(
            metric_cols=["G"],
            group_cols=["layer", "level_from", "level_to"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, "compare_step_consistency.csv"), index=False)

        Comparison(
            alt_csv_path=os.path.join(in_dir, "axis_consistency_bootstrap.csv"),
            null_csv_path=os.path.join(in_dir, "null_axis_consistency_bootstrap.csv"),
        ).compare_bootstrap(
            metric_cols=["G"],
            group_cols=["layer"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, "compare_axis_consistency.csv"), index=False)

        Comparison(
            alt_csv_path=os.path.join(in_dir, "pca_metrics_bootstrap.csv"),
            null_csv_path=os.path.join(in_dir, "null_pca_metrics_bootstrap.csv"),
        ).compare_bootstrap(
            metric_cols=[
                *[f"pc{i}" for i in range(1, self.n_pc + 1)],
                *[f"k{int(k)}" for k in self.k_list],
            ],
            group_cols=["layer"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, "compare_pca_metrics.csv"), index=False)

        Comparison(
            alt_csv_path=os.path.join(in_dir, f"pca_angles_bootstrap_k{self.k_rotation}.csv"),
            null_csv_path=os.path.join(in_dir, f"null_pca_angles_bootstrap_k{self.k_rotation}.csv"),
        ).compare_bootstrap(
            metric_cols=["mean_angle", "max_angle"],
            group_cols=["layer_from", "layer_to"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, f"compare_pca_angles_k{self.k_rotation}.csv"), index=False)

        Comparison(
            alt_csv_path=os.path.join(in_dir, f"pca_angles_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            null_csv_path=os.path.join(in_dir, f"null_pca_angles_bootstrap_var{int(self.variance_threshold*100)}.csv"),
        ).compare_bootstrap(
            metric_cols=["mean_angle", "max_angle", "k_used"],
            group_cols=["layer_from", "layer_to"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, f"compare_pca_angles_var{int(self.variance_threshold*100)}.csv"), index=False)

        Comparison(
            alt_csv_path=os.path.join(in_dir, f"pca_procrustes_bootstrap_k{self.k_rotation}.csv"),
            null_csv_path=os.path.join(in_dir, f"null_pca_procrustes_bootstrap_k{self.k_rotation}.csv"),
        ).compare_bootstrap(
            metric_cols=["residual_fro"],
            group_cols=["layer_from", "layer_to"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, f"compare_pca_procrustes_k{self.k_rotation}.csv"), index=False)

        Comparison(
            alt_csv_path=os.path.join(in_dir, f"pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            null_csv_path=os.path.join(in_dir, f"null_pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.csv"),
        ).compare_bootstrap(
            metric_cols=["residual_fro", "k_used"],
            group_cols=["layer_from", "layer_to"],
            n_boot=self.n_boot_compare,
        ).to_csv(os.path.join(out_dir, f"compare_pca_procrustes_var{int(self.variance_threshold*100)}.csv"), index=False)

    def run_plots(self, subdir=None):

        print("Running plots...")

        out_dir = self.plots_dir if subdir is None else self._out_dir(subdir)
        plots = Plots()
        plots.plot_axis_consistency(
            csv_path=os.path.join(self.stats_dir, "axis_consistency.csv"),
            output_path=os.path.join(out_dir, "axis_consistency.png"),
            title_suffix="Jackknife",
        )
        plots.plot_step_consistency(
            csv_path=os.path.join(self.stats_dir, "step_consistency.csv"),
            output_path=os.path.join(out_dir, "step_consistency.png"),
            title_suffix="Jackknife",
        )
        plots.plot_pca_components(
            components=[1, 2, 3],
            cumulative=True,
            csv_path=os.path.join(self.pca_dir, "pca_metrics_jackknife.csv"),
            output_path=os.path.join(out_dir, "pca_components.png"),
            title_suffix="Jackknife",
        )
        plots.plot_pca_k_thresholds(
            thresholds=[85, 90, 95],
            csv_path=os.path.join(self.pca_dir, "pca_metrics_jackknife.csv"),
            output_path=os.path.join(out_dir, "pca_thresholds.png"),
            title_suffix="Jackknife",
        )
        plots.plot_principal_angles(
            csv_path=os.path.join(self.pca_dir, f"pca_angles_k{self.k_rotation}.csv"),
            output_path=os.path.join(out_dir, f"pca_angles_k{self.k_rotation}.png"),
            title_suffix="Jackknife",
        )
        plots.plot_principal_angles(
            csv_path=os.path.join(self.pca_dir, f"pca_angles_var{int(self.variance_threshold*100)}.csv"),
            output_path=os.path.join(out_dir, f"pca_angles_var{int(self.variance_threshold*100)}.png"),
            title_suffix="Jackknife",
        )
        plots.plot_procrustes_alignment(
            csv_path=os.path.join(self.pca_dir, f"pca_procrustes_k{self.k_rotation}.csv"),
            output_path=os.path.join(out_dir, f"pca_procrustes_k{self.k_rotation}.png"),
            title_suffix="Jackknife",
        )
        plots.plot_procrustes_alignment(
            csv_path=os.path.join(self.pca_dir, f"pca_procrustes_var{int(self.variance_threshold*100)}.csv"),
            output_path=os.path.join(out_dir, f"pca_procrustes_var{int(self.variance_threshold*100)}.png"),
            title_suffix="Jackknife",
        )

        plots.plot_axis_consistency(
            csv_path=os.path.join(self.bootstrap_dir, "axis_consistency_bootstrap.csv"),
            output_path=os.path.join(out_dir, "axis_consistency_bootstrap.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_step_consistency(
            csv_path=os.path.join(self.bootstrap_dir, "step_consistency_bootstrap.csv"),
            output_path=os.path.join(out_dir, "step_consistency_bootstrap.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_pca_components(
            components=[1, 2, 3],
            cumulative=True,
            csv_path=os.path.join(self.bootstrap_dir, "pca_metrics_bootstrap.csv"),
            output_path=os.path.join(out_dir, "pca_components_bootstrap.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_pca_k_thresholds(
            thresholds=[85, 90, 95],
            csv_path=os.path.join(self.bootstrap_dir, "pca_metrics_bootstrap.csv"),
            output_path=os.path.join(out_dir, "pca_thresholds_bootstrap.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_principal_angles(
            csv_path=os.path.join(self.bootstrap_dir, f"pca_angles_bootstrap_k{self.k_rotation}.csv"),
            output_path=os.path.join(out_dir, f"pca_angles_bootstrap_k{self.k_rotation}.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_principal_angles(
            csv_path=os.path.join(self.bootstrap_dir, f"pca_angles_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            output_path=os.path.join(out_dir, f"pca_angles_bootstrap_var{int(self.variance_threshold*100)}.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_procrustes_alignment(
            csv_path=os.path.join(self.bootstrap_dir, f"pca_procrustes_bootstrap_k{self.k_rotation}.csv"),
            output_path=os.path.join(out_dir, f"pca_procrustes_bootstrap_k{self.k_rotation}.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )
        plots.plot_procrustes_alignment(
            csv_path=os.path.join(self.bootstrap_dir, f"pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            output_path=os.path.join(out_dir, f"pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.png"),
            title_suffix="Bootstrap",
            percentile_ci=True,
        )

        plots.plot_axis_consistency(
            csv_path=os.path.join(self.bootstrap_dir, "null_axis_consistency_bootstrap.csv"),
            output_path=os.path.join(out_dir, "null_axis_consistency_bootstrap.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_step_consistency(
            csv_path=os.path.join(self.bootstrap_dir, "null_step_consistency_bootstrap.csv"),
            output_path=os.path.join(out_dir, "null_step_consistency_bootstrap.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_pca_components(
            components=[1, 2, 3],
            cumulative=True,
            csv_path=os.path.join(self.bootstrap_dir, "null_pca_metrics_bootstrap.csv"),
            output_path=os.path.join(out_dir, "null_pca_components_bootstrap.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_pca_k_thresholds(
            thresholds=[85, 90, 95],
            csv_path=os.path.join(self.bootstrap_dir, "null_pca_metrics_bootstrap.csv"),
            output_path=os.path.join(out_dir, "null_pca_thresholds_bootstrap.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_principal_angles(
            csv_path=os.path.join(self.bootstrap_dir, f"null_pca_angles_bootstrap_k{self.k_rotation}.csv"),
            output_path=os.path.join(out_dir, f"null_pca_angles_bootstrap_k{self.k_rotation}.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_principal_angles(
            csv_path=os.path.join(self.bootstrap_dir, f"null_pca_angles_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            output_path=os.path.join(out_dir, f"null_pca_angles_bootstrap_var{int(self.variance_threshold*100)}.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_procrustes_alignment(
            csv_path=os.path.join(self.bootstrap_dir, f"null_pca_procrustes_bootstrap_k{self.k_rotation}.csv"),
            output_path=os.path.join(out_dir, f"null_pca_procrustes_bootstrap_k{self.k_rotation}.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )
        plots.plot_procrustes_alignment(
            csv_path=os.path.join(self.bootstrap_dir, f"null_pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.csv"),
            output_path=os.path.join(out_dir, f"null_pca_procrustes_bootstrap_var{int(self.variance_threshold*100)}.png"),
            title_suffix="Null Bootstrap",
            percentile_ci=True,
        )


def main():
    gpt = GPT2()
    sentiment_template = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    null_sentiment_template = Template(SENTIMENT_SENTENCES, NULL_WORDS)
    pipeline = Pipeline(
        output_path="outputs",
        k_list=[85, 90, 95],
        stats_subdir="data/stats",
        pca_subdir="data/pca",
        bootstrap_subdir="data/bootstrap",
        compare_subdir="data/comparison",
        plots_subdir="plots",
    )

    sentiment_deltas = pipeline.get_deltas(sentiment_template, gpt)
    null_sentiment_deltas = pipeline.get_deltas(null_sentiment_template, gpt)

    pipeline.get_stats(sentiment_deltas)
    pipeline.run_pca(sentiment_deltas)
    pipeline.run_bootstrap(sentiment_deltas, null_sentiment_deltas)
    pipeline.run_comparison()
    pipeline.run_plots()

if __name__ == "__main__":
    main()
