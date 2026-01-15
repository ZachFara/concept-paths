import os
import re
import sys
from typing import List, Optional

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.templates import Template, SENTIMENT_SENTENCES, SENTIMENT_WORDS
from src.capture import GPT2
from src.logs import setup_logger

logger = setup_logger(__name__)

class Deltas:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df["level"] = self.df["level_id"].apply(self.level_to_int)
        self.ensure_hidden_last()

    @staticmethod
    def level_to_int(value) -> Optional[int]:
        match = re.search(r"(\d+)", str(value))
        return int(match.group(1)) if match else None

    @staticmethod
    def pool_last_from_padded(row) -> torch.Tensor:
        resid = row["padded_residual"]
        idx = max(int(row["seq_len"]) - 1, 0)
        return resid[:, idx, :].squeeze(0)

    def ensure_hidden_last(self) -> None:
        if "hidden_last" in self.df.columns:
            return
        self.df["hidden_last"] = self.df.apply(self.pool_last_from_padded, axis=1)

    @staticmethod
    def mean_tensor(series: pd.Series) -> torch.Tensor:
        return torch.stack(series.tolist(), dim=0).mean(dim=0)

    def compute_mu(self, group_cols: List[str]) -> pd.DataFrame:
        return (
            self.df.groupby(group_cols + ["level"], sort=True)["hidden_last"]
            .apply(self.mean_tensor)
            .reset_index()
            .rename(columns={"hidden_last": "mu"})
        )

    def compute_adjacent_deltas(
        self, mu: pd.DataFrame, group_cols: List[str]
    ) -> pd.DataFrame:
        rows = []
        for group_key, group in mu.groupby(group_cols, sort=False):
            group = group.sort_values("level")
            levels = group["level"].tolist()
            for i, level in enumerate(levels):
                if i + 1 >= len(levels):
                    continue
                next_level = levels[i + 1]
                if next_level != level + 1:
                    continue
                mu_l = group.iloc[i]["mu"]
                mu_next = group.iloc[i + 1]["mu"]
                row = {
                    "level_from": level,
                    "level_to": next_level,
                    "delta": mu_next - mu_l,
                }
                if isinstance(group_key, tuple):
                    for col, value in zip(group_cols, group_key):
                        row[col] = value
                else:
                    row[group_cols[0]] = group_key
                rows.append(row)
        return pd.DataFrame(rows)

    def cache_deltas(self, df, path):

        df.to_csv(path)

        return None

    def load_deltas(self, path):

        df = pd.read_csv(path)

        return df

def main():
    per_layer = True  # set False to use only last layer
    output_path = os.path.join(REPO_ROOT, "deltas_adjacent.pkl")

    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()

    df = temp.get_grid_sentences(
        sentence_ids=[1, 2],
        level_ids=[1, 2],
        word_ids=[1, 2, 3],
    )

    if per_layer:
        df = gpt.add_x_residuals_to_df(df)
        group_cols = ["sentence_id", "layer"]
    else:
        df = gpt.add_padded_residuals_to_df(df)
        group_cols = ["sentence_id"]

    deltas = Deltas(df)

    mu = deltas.compute_mu(group_cols)
    deltas_adj = deltas.compute_adjacent_deltas(mu, group_cols)

    deltas_adj.to_pickle(output_path)

    hidden_unique = deltas.df["hidden_last"].nunique(dropna=False)
    print(f"mu rows: {len(mu)}")
    print(f"adjacent deltas rows: {len(deltas_adj)}")
    print(f"hidden_last unique count: {hidden_unique} / {len(deltas.df)}")
    if not deltas_adj.empty:
        print(f"delta tensor shape example: {tuple(deltas_adj['delta'].iloc[0].shape)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
