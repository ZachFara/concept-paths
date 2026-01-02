from typing import List, Dict, Optional
import argparse
import os
import math
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.capture import GPT2
from src.deltas import Deltas
from src.logs import setup_logger
from src.pca import PCA
from src.templates import SENTIMENT_SENTENCES, SENTIMENT_WORDS, SENTIMENT_ABLATION_TEMPLATE, Template
from src.deltas import Deltas
from src.pca import PCA
from src.ablate.ablationdata import AblationData
from src.ablate.ablators import GPT2Ablator
from src.ablate.ablationresults import AblationResults
from src.config import Config

logger = setup_logger(__name__)

class SeedController:

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        top_ks: Optional[List[int]] = None,
        ablator_cls=GPT2Ablator,
        config: Optional[Config] = None,
        seed: int = 0,
    ):
        if seeds is None:
            if config is not None and getattr(config, "random_seeds", None) is not None:
                seeds = list(config.random_seeds)
            else:
                seeds = [seed]
        self.seeds = [int(s) for s in seeds]
        self.top_ks = top_ks or []
        self.ablator_cls = ablator_cls
        self.config = config
        self.seed = seed
        self.results_by_seed: Dict[int, AblationResults] = {}

    def _infer_methods(self, df: pd.DataFrame) -> List[str]:
        methods = []
        for col in df.columns:
            if col.endswith("_pos") and f"{col[:-4]}_neg" in df.columns:
                methods.append(col[:-4])
        return sorted(set(methods))

    def run(
        self,
        data: AblationData,
        deltas_adj: pd.DataFrame,
        templated_sentences_df: Optional[pd.DataFrame] = None,
    ) -> Dict[int, AblationResults]:

        for seed in self.seeds:
            logger.info("Running ablations for seed=%s", seed)
            if templated_sentences_df is None:
                seed_df = data.get_templated_sentences(seed=seed)
            else:
                seed_df = templated_sentences_df
            ablator_factory = lambda d, k, s=seed: self.ablator_cls(d, k, seed=s)
            results = AblationResults(
                top_ks=self.top_ks,
                ablator=ablator_factory,
                seed=seed,
            )
            results.gather_results(data=data, deltas=deltas_adj, df=seed_df)
            self.results_by_seed[seed] = results

        return self.results_by_seed

    def summarize_over_k_across_seeds(
        self,
        metric="nll",
        n_boot=1000,
        seed: Optional[int] = None,
        eps=1e-9,
    ) -> pd.DataFrame:
        if not self.results_by_seed:
            raise ValueError("No results available to summarize across seeds")
        if seed is None:
            seed = self.seed

        seeds = sorted(self.results_by_seed.keys())
        first_results = self.results_by_seed[seeds[0]]
        if not first_results.results:
            raise ValueError("No per-k results found for the first seed")

        methods = self._infer_methods(next(iter(first_results.results.values())))
        k_values = sorted(first_results.results.keys())

        per_seed = {s: {} for s in seeds}
        for s in seeds:
            results = self.results_by_seed[s]
            for k in k_values:
                df = results.results.get(k)
                if df is None:
                    continue
                is_pos = (df["label"] == "positive").values
                for method in methods:
                    pos = df[f"{method}_pos"].values
                    neg = df[f"{method}_neg"].values
                    correct_prob = torch.tensor(
                        [p if pos_flag else n for p, n, pos_flag in zip(pos, neg, is_pos)],
                        dtype=torch.float32,
                    ).clamp(min=eps)
                    if metric == "accuracy":
                        sample_scores = (torch.tensor(pos) >= torch.tensor(neg)).float()
                    elif metric == "nll":
                        sample_scores = -torch.log(correct_prob)
                    else:
                        raise ValueError("metric must be 'nll' or 'accuracy'")
                    per_seed[s][(k, method)] = sample_scores.mean().item()

        rng = torch.Generator().manual_seed(int(seed))
        rows = []
        for k in k_values:
            for method in methods:
                vals = [
                    per_seed[s][(k, method)]
                    for s in seeds
                    if (k, method) in per_seed[s]
                ]
                if not vals:
                    continue
                values = torch.tensor(vals, dtype=torch.float32)
                obs_mean = values.mean().item()
                boot_means = []
                n = len(values)
                for _ in range(int(n_boot)):
                    idx = torch.randint(0, n, (n,), generator=rng)
                    boot_means.append(values[idx].mean().item())
                boot_means = sorted(boot_means)
                lo = boot_means[int(0.025 * len(boot_means))]
                hi = boot_means[int(0.975 * len(boot_means))]
                rows.append(
                    {
                        "top_k": k,
                        "method": method,
                        "metric": metric,
                        "mean": obs_mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_seeds": len(values),
                    }
                )
        return pd.DataFrame(rows)

    def plot_over_k_across_seeds(
        self,
        metric="nll",
        n_boot=1000,
        seed: Optional[int] = None,
        eps=1e-9,
        output_path: Optional[str] = None,
        title_suffix: Optional[str] = None,
    ):
        summary = self.summarize_over_k_across_seeds(
            metric=metric,
            n_boot=n_boot,
            seed=seed,
            eps=eps,
        )
        fig, ax = plt.subplots()
        for method in sorted(summary["method"].unique()):
            g = summary[summary["method"] == method].sort_values("top_k")
            ax.plot(g["top_k"], g["mean"], marker="o", label=method)
            ax.fill_between(g["top_k"], g["ci_low"], g["ci_high"], alpha=0.2)

        ax.set_xlabel("Top-k")
        ylabel = "Mean NLL" if metric == "nll" else "Accuracy"
        ax.set_ylabel(ylabel)
        title = "Ablation Performance over k (seed-averaged)"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend()

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax


def main():
    cfg = Config("config/test.yaml")
    top_ks = list(cfg.get("top_ks", []))
    seeds = list(cfg.get("random_seeds", [cfg.get("random_seed", 0)]))
    train_split = cfg.get("train_split", 0.8)
    probe_epochs = cfg.get("probe_epochs", 100)
    n_boot = cfg.get("n_boot", 1000)
    paths = cfg.get("paths", None)
    out_dir = paths.outputs if paths else "outputs/ablation_results"

    data = AblationData(
        SENTIMENT_SENTENCES,
        SENTIMENT_WORDS,
        4,
        SENTIMENT_ABLATION_TEMPLATE,
        config=cfg,
    )
    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()
    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df=df, x=None)
    delta = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)

    controller = SeedController(
        seeds=seeds,
        top_ks=top_ks,
        ablator_cls=GPT2Ablator,
        config=cfg,
    )
    results_by_seed = controller.run(
        data=data,
        deltas_adj=deltas_adj,
    )

    os.makedirs(out_dir, exist_ok=True)
    summary_nll = controller.summarize_over_k_across_seeds(
        metric="nll", n_boot=n_boot
    )
    summary_acc = controller.summarize_over_k_across_seeds(
        metric="accuracy", n_boot=n_boot
    )
    summary_nll.to_csv(os.path.join(out_dir, "seed_avg_performance_nll.csv"), index=False)
    summary_acc.to_csv(
        os.path.join(out_dir, "seed_avg_performance_accuracy.csv"), index=False
    )
    controller.plot_over_k_across_seeds(
        metric="nll",
        n_boot=n_boot,
        output_path=os.path.join(out_dir, "seed_avg_performance_nll.png"),
    )
    controller.plot_over_k_across_seeds(
        metric="accuracy",
        n_boot=n_boot,
        output_path=os.path.join(out_dir, "seed_avg_performance_accuracy.png"),
    )

    for seed, results in results_by_seed.items():
        if not results.results:
            continue
        first_k = results.top_ks[0]
        scored_df = results.results[first_k]
        best_summary = results.best_method_summary(df=scored_df)
        best_summary.to_csv(
            os.path.join(out_dir, f"best_method_summary_seed_{seed}.csv"),
            index=False,
        )
        logger.info("Best method summary for seed=%s:\n%s", seed, best_summary)

    print(summary_nll)
    print(summary_acc)


if __name__ == "__main__":
    main()
