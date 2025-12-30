import itertools
import torch
import pandas as pd
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capture import GPT2
from src.templates import SENTIMENT_WORDS, SENTIMENT_SENTENCES, Template
from src.deltas import Deltas

def cos(a, b):
  return torch.nn.functional.cosine_similarity(a, b, dim=0)

class Stats:

    def __init__(self, adjacent_deltas_df):
        self.adjacent_deltas_df = adjacent_deltas_df
    
    def step_consistency(self, df = None):

        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()

    # Analysis A: For each step i.e. (1-2) and layer combination, get the cos similarity of all sentenc IDs and return the average cos similarity for that combination

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

        return analysisA_df

    def axis_consistency(self, df = None):

        if df is None:
            deltas_adj = self.adjacent_deltas_df.copy()
        else:
            deltas_adj = df.copy()
        
        # Analysis B: For each layer, average the steps (1 - 2) to get a single direction per layer and sentence ID combination. Then compute the pairwise cos similarity between each sentence ID and return the average for that layer
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

        return analysisB_df

    def jackknife_step_consistency(self, df=None, id_col="sentence_id"):
        deltas_adj = self.adjacent_deltas_df.copy() if df is None else df.copy()
        results = []
        for item_id in deltas_adj[id_col].unique():
            subset = deltas_adj[deltas_adj[id_col] != item_id]
            analysis = self.step_consistency(df=subset)
            if analysis.empty:
                continue
            analysis = analysis.copy()
            analysis["left_out"] = item_id
            results.append(analysis)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def jackknife_axis_consistency(self, df=None, id_col="sentence_id"):
        deltas_adj = self.adjacent_deltas_df.copy() if df is None else df.copy()
        results = []
        for item_id in deltas_adj[id_col].unique():
            subset = deltas_adj[deltas_adj[id_col] != item_id]
            analysis = self.axis_consistency(df=subset)
            if analysis.empty:
                continue
            analysis = analysis.copy()
            analysis["left_out"] = item_id
            results.append(analysis)
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)


def main():

    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()

    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df = df, x = None)

    delta = Deltas(df)

    group_cols = ["sentence_id", "layer"]

    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)

    stats = Stats(adjacent_deltas_df= deltas_adj)

    axis_cons = stats.jackknife_axis_consistency()
    step_cons = stats.jackknife_step_consistency()

    axis_cons.to_csv("outputs/axis_consistency.csv")
    step_cons.to_csv("outputs/step_consistency.csv")

if __name__ == "__main__":
    main()
