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

logger = setup_logger(__name__)

class AblationResults:

    # TODO: Further make it so that we can test multiple random seeds and take the average over those random seeds to reduce the variance.

    def __init__(
        self,
        top_ks: Optional[List[int]] = None,
        results: Optional[Dict[int, pd.DataFrame]] = None,
        ablator: GPT2Ablator = None,
        df: Optional[pd.DataFrame] = None,
    ):

        # TODO: Make the function signature use a different type hint for ablator. It should be something more general

        self.top_ks = top_ks or []
        self.results = results or {}
        self.ablator = ablator
        self.df = df
        if self.ablator is None:
            logger.info("AblationResults instance loaded without ablator. This will cause a crash if we attempt to gather the results without an ablator")

    def gather_results(self, data:AblationData, deltas, df):


        # Get the PC's once
        pca = PCA(deltas)
        pca_dict = pca.get_all_layer_pca(deltas)

        # Get all of the results for each of the self.ks
        for k in self.top_ks:

            logger.info(f"Running ablation for k = {k}")

            current_ablator = self.ablator(data, k)
            # Train the linear probes
            current_ablator.train_linear_probes(df)

            result_for_k = current_ablator.fill_test_df(df, pca_dict)

            self.results[k] = result_for_k

    def _get_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is not None:
            return df
        if self.df is None:
            raise ValueError("No dataframe provided to AblationResults")
        return self.df

    def _infer_methods(self, df: pd.DataFrame) -> List[str]:
        methods = []
        for col in df.columns:
            if col.endswith("_pos") and f"{col[:-4]}_neg" in df.columns:
                methods.append(col[:-4])
        return sorted(set(methods))

    def score_two_methods(
        self,
        method_a="linear",
        method_b="random",
        n_boot=1000,
        n_perm=1000,
        seed=0,
        eps=1e-9,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = self._get_df(df).copy()

        def binom_cdf(k, n, p):
            return sum(
                math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
                for i in range(k + 1)
            )

        a_pos = df[f"{method_a}_pos"].values
        a_neg = df[f"{method_a}_neg"].values
        b_pos = df[f"{method_b}_pos"].values
        b_neg = df[f"{method_b}_neg"].values
        is_pos = (df["label"] == "positive").values

        a_prob = torch.tensor(
            [p if pos else n for p, n, pos in zip(a_pos, a_neg, is_pos)],
            dtype=torch.float32,
        ).clamp(min=eps)
        b_prob = torch.tensor(
            [p if pos else n for p, n, pos in zip(b_pos, b_neg, is_pos)],
            dtype=torch.float32,
        ).clamp(min=eps)

        delta = (-torch.log(b_prob)) - (-torch.log(a_prob))
        obs_mean = delta.mean().item()

        rng = torch.Generator().manual_seed(int(seed))
        boot_means = []
        n = len(delta)
        for _ in range(int(n_boot)):
            idx = torch.randint(0, n, (n,), generator=rng)
            boot_means.append(delta[idx].mean().item())
        boot_means = sorted(boot_means)
        lo = boot_means[int(0.025 * len(boot_means))]
        hi = boot_means[int(0.975 * len(boot_means))]

        perm_means = []
        for _ in range(int(n_perm)):
            signs = torch.randint(0, 2, (n,), generator=rng) * 2 - 1
            perm_means.append((delta * signs).mean().item())
        perm_means = torch.tensor(perm_means)
        perm_p = (perm_means.abs() >= abs(obs_mean)).float().mean().item()

        a_correct = (a_pos >= a_neg) == is_pos
        b_correct = (b_pos >= b_neg) == is_pos
        wins_a = int(((a_correct == True) & (b_correct == False)).sum())
        wins_b = int(((a_correct == False) & (b_correct == True)).sum())
        n_wins = wins_a + wins_b

        if n_wins == 0:
            sign_p = 1.0
        else:
            k = min(wins_a, wins_b)
            sign_p = 2 * binom_cdf(k, n_wins, 0.5)
            sign_p = min(1.0, sign_p)

        return pd.DataFrame(
            [
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_nll_gap": obs_mean,
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                    "perm_p_value": perm_p,
                    "sign_test_p_value": sign_p,
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                }
            ]
        )

    def best_method_per_sample(
        self,
        df: Optional[pd.DataFrame] = None,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = self._get_df(df).copy()
        methods = methods or self._infer_methods(df)
        if not methods:
            raise ValueError("No method columns found for best_method_per_sample")

        is_pos = (df["label"] == "positive").values
        scores = []
        for method in methods:
            pos = df[f"{method}_pos"].values
            neg = df[f"{method}_neg"].values
            scores.append([p if pos_flag else n for p, n, pos_flag in zip(pos, neg, is_pos)])

        score_matrix = torch.tensor(scores).T
        best_idx = torch.argmax(score_matrix, dim=1).tolist()
        best_methods = [methods[i] for i in best_idx]
        best_scores = score_matrix.max(dim=1).values.tolist()

        result = df.copy()
        result["best_method"] = best_methods
        result["best_score"] = best_scores
        return result

    def score_all_pairs(
        self,
        df: Optional[pd.DataFrame] = None,
        methods: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        df = self._get_df(df)
        methods = methods or self._infer_methods(df)
        rows = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                rows.append(
                    self.score_two_methods(
                        methods[i],
                        methods[j],
                        df=df,
                        **kwargs,
                    ).iloc[0]
                )
        return pd.DataFrame(rows)

    def best_method_summary(
        self,
        df: Optional[pd.DataFrame] = None,
        methods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        best_df = self.best_method_per_sample(df=df, methods=methods)
        counts = best_df["best_method"].value_counts().reset_index()
        counts.columns = ["method", "wins"]
        counts["win_rate"] = counts["wins"] / counts["wins"].sum()
        return counts

    def summarize_over_k(
        self,
        metric="nll",
        methods: Optional[List[str]] = None,
        n_boot=1000,
        seed=0,
        eps=1e-9,
    ) -> pd.DataFrame:
        if not self.results:
            raise ValueError("No results available to summarize over k")
        rng = torch.Generator().manual_seed(int(seed))
        rows = []
        for k in sorted(self.results.keys()):
            df = self.results[k]
            methods = methods or self._infer_methods(df)
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

                obs_mean = sample_scores.mean().item()
                boot_means = []
                n = len(sample_scores)
                for _ in range(int(n_boot)):
                    idx = torch.randint(0, n, (n,), generator=rng)
                    boot_means.append(sample_scores[idx].mean().item())
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
                    }
                )
        return pd.DataFrame(rows)

    def plot_over_k(
        self,
        metric="nll",
        methods: Optional[List[str]] = None,
        n_boot=1000,
        seed=0,
        eps=1e-9,
        output_path: Optional[str] = None,
        title_suffix: Optional[str] = None,
    ):
        summary = self.summarize_over_k(
            metric=metric,
            methods=methods,
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
        title = "Ablation Performance over k"
        if title_suffix:
            title = f"{title} ({title_suffix})"
        ax.set_title(title)
        ax.legend()

        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig, ax

def main():

    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 4, SENTIMENT_ABLATION_TEMPLATE)
    templated_sentences_df = data.get_templated_sentences(train_split = .8)

    ks = [0, 5, 10, 20, 40, 60, 80, 100]

    results = AblationResults(df = templated_sentences_df, top_ks = ks, ablator=GPT2Ablator)

    # Gather the deltas
    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()
    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df = df, x = None)
    delta = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)

    results.gather_results(deltas = deltas_adj, data = data, df = templated_sentences_df)

    if not results.results:
        raise ValueError("No scored results were produced")
    first_k = results.top_ks[0]
    scored_df = results.results[first_k]

    best_df = results.best_method_per_sample(df=scored_df)
    best_summary = results.best_method_summary(df=scored_df)
    pairwise = results.score_all_pairs(df=scored_df)

    for k in results.top_ks:
        if k not in results.results:
            logger.warning("No results found for k=%s", k)
            continue
        k_summary = results.best_method_summary(df=results.results[k])
        logger.info("Best method summary for k=%s:\n%s", k, k_summary)

    out_dir = "outputs/ablation_results"
    os.makedirs(out_dir, exist_ok=True)
    best_df.to_csv(os.path.join(out_dir, "best_method_per_sample.csv"), index=False)
    best_summary.to_csv(os.path.join(out_dir, "best_method_summary.csv"), index=False)
    pairwise.to_csv(os.path.join(out_dir, "pairwise_significance.csv"), index=False)

    if len(results.results) > 1:
        summary_nll = results.summarize_over_k(metric="nll")
        summary_acc = results.summarize_over_k(metric="accuracy")
        summary_nll.to_csv(
            os.path.join(out_dir, "performance_over_k_nll.csv"), index=False
        )
        summary_acc.to_csv(
            os.path.join(out_dir, "performance_over_k_accuracy.csv"), index=False
        )
        results.plot_over_k(
            metric="nll",
            output_path=os.path.join(out_dir, "performance_over_k_nll.png"),
        )
        results.plot_over_k(
            metric="accuracy",
            output_path=os.path.join(out_dir, "performance_over_k_accuracy.png"),
        )

    print(best_summary)
    print(pairwise)

if __name__ == "__main__":
    main()
