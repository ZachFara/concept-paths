"""
This script is meant to use the modules within src to test that between different templates is our cosine similarity high or low?

If it is high that would indicate that our deltas really do capture the same "sentiment direction" between different sentence templates.
"""
import os
import sys
import torch
import itertools
import pandas as pd
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capture import GPT2
from src.templates import SENTIMENT_WORDS, SENTIMENT_SENTENCES, Template
from src.deltas import Deltas

def cos(a, b):
  return torch.nn.functional.cosine_similarity(a, b, dim=0)

def main():

    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()

    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df = df, x = None)

    delta = Deltas(df)

    group_cols = ["sentence_id", "layer"]

    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)

    print(f"Adjacent Deltas (Shape: {deltas_adj.shape} Column: {deltas_adj.columns}):\n{deltas_adj}")

    # Get out cosin similarities
    # Analysis A: Compare templates within each adjacent step, then average

    analysisA = []

    for (layer, level_from, level_to), g in deltas_adj.groupby(['layer', 'level_from', 'level_to']):
        deltas = g.set_index("sentence_id")['delta'].to_dict()

        pairs = list(itertools.combinations(deltas.keys(), 2))

        if not pairs:
            continue
        sims = []

        for t1, t2 in pairs:
            sims.append(cos(deltas[t1], deltas[t2]).item())

        analysisA.append(
            {
                "layer": layer,
                "level_from": level_from,
                "level_to": level_to,
                "G": sum(sims) / len(sims),
            }
        )

    analysisA_df = pd.DataFrame(analysisA)
    analysisA_df.to_csv("outputs/analysisA.csv")

    # Analysis B: Aggregate steps into one “template direction” per layer, then compare
    analysisB = []
    template_vectors = []
    for (sentence_id, layer), g in deltas_adj.groupby(["sentence_id", "layer"]):
        v = torch.stack(list(g["delta"])).mean(dim=0)
        template_vectors.append(
            {"sentence_id": sentence_id, "layer": layer, "v": v}
        )
    template_vectors = pd.DataFrame(template_vectors)

    for layer, g in template_vectors.groupby("layer"):
        vectors = g.set_index("sentence_id")["v"].to_dict()
        pairs = list(itertools.combinations(vectors.keys(), 2))
        if not pairs:
            continue
        sims = [cos(vectors[t1], vectors[t2]).item() for t1, t2 in pairs]
        analysisB.append({"layer": layer, "G": sum(sims) / len(sims)})
    
    analysisB_df = pd.DataFrame(analysisB)
    analysisB_df.to_csv("outputs/analysisB.csv")

if __name__ == "__main__":
    main()
