import math
import pandas as pd
import torch

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

    def __init__(self, df):
        self.df = df.copy()

    def score_significance(
        self,
        method_a="linear",
        method_b="random",
        n_boot=1000,
        n_perm=1000,
        seed=0,
        eps=1e-9,
        all_pairs=False,
    ):
        df = self.df.copy()

        def binom_cdf(k, n, p):
            return sum(
                math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
                for i in range(k + 1)
            )

        def score_pair(a_name, b_name):
            a_pos = df[f"{a_name}_pos"].values
            a_neg = df[f"{a_name}_neg"].values
            b_pos = df[f"{b_name}_pos"].values
            b_neg = df[f"{b_name}_neg"].values
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

            return {
                "method_a": a_name,
                "method_b": b_name,
                "mean_nll_gap": obs_mean,
                "boot_ci_low": lo,
                "boot_ci_high": hi,
                "perm_p_value": perm_p,
                "sign_test_p_value": sign_p,
                "wins_a": wins_a,
                "wins_b": wins_b,
            }

        if all_pairs:
            methods = []
            for col in df.columns:
                if col.endswith("_pos") and f"{col[:-4]}_neg" in df.columns:
                    methods.append(col[:-4])
            methods = sorted(set(methods))
            rows = []
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    rows.append(score_pair(methods[i], methods[j]))
            return pd.DataFrame(rows)

        return pd.DataFrame([score_pair(method_a, method_b)])

def score_df():
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 4)
    ablator = GPT2Ablator(data, SENTIMENT_ABLATION_TEMPLATE, 10)
    templated_sentences_df = data.get_templated_sentences(train_split = .8)
    ablator.train_linear_probes(templated_sentences_df, epochs=100)
    
    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()

    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df = df, x = None)

    delta = Deltas(df)

    group_cols = ["sentence_id", "layer"]

    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)

    pca = PCA(deltas_adj)

    pca_dict = pca.get_all_layer_pca(deltas_adj)

    print("Getting Scored DF")
    scored_df = ablator.fill_test_df(templated_sentences_df, pca_dict)
    results = AblationResults(scored_df)
    scored_summary = ablator.score_probas_df(scored_df)
    scored_significance = results.score_significance(all_pairs=True)
    print(scored_summary)
    print(scored_significance)
    scored_summary.to_csv("summary.csv")
    scored_significance.to_csv("significance.csv")


def main():
    score_df()

if __name__ == "__main__":
    main()
