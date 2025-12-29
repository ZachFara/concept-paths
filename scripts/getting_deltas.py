import pandas as pd
import os
import re
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.templates import Template, SENTIMENT_SENTENCES, SENTIMENT_WORDS
from src.capture import GPT2

def pool_last(row):
    resid = row["padded_residual"]   # [L, T, D]
    idx = max(row["seq_len"] - 1, 0)
    return resid[:, idx, :]          # [L, D]

def extract_level(value):
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None

def adjacent_deltas(g):
    g = g.sort_values("level")
    g["mu_next"] = g["mu"].shift(-1)
    g["level_next"] = g["level"].shift(-1)

    g = g[g["level_next"] == g["level"] + 1]

    return pd.DataFrame({
        "sentence_id": g["sentence_id"],
        "level_from": g["level"],
        "level_to": g["level_next"],
        "delta": g["mu_next"].values - g["mu"].values,
    })



def main():

    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()

    df = temp.get_grid_sentences(
            sentence_ids=[1],
            level_ids=[1,2],
            word_ids=[1,2,3]
            )

    print(df.head(3))

    df = gpt.add_padded_residuals_to_df(df)
    
    print(df.head(1))
    print(df.columns)

    df.to_pickle("testing.pkl")

    df["level"] = df["level_id"].apply(extract_level)

    df["h_last"] = df.apply(pool_last, axis=1)

    mu = (
        df.groupby(["sentence_id", "level"], sort=True)["h_last"]
          .apply(lambda xs: torch.stack(xs.tolist(), dim=0).mean(dim=0))  # [L, D]
          .reset_index()
          .rename(columns={"h_last": "mu"})
    )

    deltas_adj = (
        mu.groupby("sentence_id", group_keys=False)
          .apply(adjacent_deltas)
          .reset_index(drop=True)
    )

    print(f"Adjacent Deltas (Shape: {deltas_adj.shape}):\n{deltas_adj}")

if __name__ == "__main__":
    main()
