from typing import List, Dict
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

    def __init__(self, top_ks:List[int], results:Dict[pd.DataFrame], ablator:GPT2Ablator = None): 

        # TODO: Make the function signature use a different type hint for ablator. It should be something more general

        self.top_ks = top_ks
        self.results = results
        self.ablator = ablator
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
            self.ablator.train_linear_probes(df)

            result_for_k = self.ablator.fill_test_df(df, pca_dict)

            self.results[k] = result_for_k


def main():
    pass

if __name__ == "__main__":
    main()
