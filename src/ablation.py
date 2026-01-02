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
from src.ablate.ablationresults import AblationResults

logger = setup_logger(__name__)


def main():
    pass

if __name__ == "__main__":
    main()
