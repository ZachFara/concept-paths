import math
import pandas as pd
import torch
from src.capture import GPT2
from src.deltas import Deltas
from src.logs import setup_logger
from src.pca import PCA
from src.templates import SENTIMENT_SENTENCES, SENTIMENT_WORDS, Template, SENTIMENT_ABLATION_TEMPLATE
from src.deltas import Deltas
from src.pca import PCA
from src.config import Config

logger = setup_logger(__name__)


class AblationData:
    def __init__(self, sentences, words, cutoff, template=None, config=None, seed=0):
        """Cutoff is the number of bottom levels treated as negative and top levels treated as positive."""
        self.sentences = sentences
        self.words = words
        self.cutoff = int(cutoff)
        self.template = template
        if config is not None:
            self.seed = config.get("random_seed", 0)
        else:
            self.seed = seed
        self._validate()

    def _validate(self):
        if self.cutoff <= 0:
            raise ValueError("cutoff must be a positive integer")
        level_keys = self._sorted_level_keys()
        if 2 * self.cutoff > len(level_keys):
            raise ValueError(
                "cutoff is too large for the number of levels (needs 2*cutoff <= levels)"
            )

    def _sorted_sentence_keys(self):
        return sorted(
            self.sentences.keys(),
            key=lambda k: int(k.split(".")[1]),
        )

    def _sorted_level_keys(self):
        return sorted(
            self.words.keys(),
            key=lambda k: int(k.split(".")[1]),
        )

    def get_level_groups(self):
        level_keys = self._sorted_level_keys()
        negative = level_keys[: self.cutoff]
        positive = level_keys[-self.cutoff :]
        return {
            "negative": negative,
            "positive": positive,
        }

    def get_labeled_sentences(self, train_split=0.8, split_by="sentence_id", seed=None):
        if seed is None:
            seed = self.seed
        groups = self.get_level_groups()
        sentence_keys = self._sorted_sentence_keys()
        if split_by != "sentence_id":
            raise ValueError("split_by must be 'sentence_id'")
        if not 0 < train_split < 1:
            raise ValueError("train_split must be between 0 and 1")

        rng = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(len(sentence_keys), generator=rng).tolist()
        split_idx = int(round(len(sentence_keys) * train_split))
        train_ids = set(sentence_keys[i] for i in indices[:split_idx])

        rows = []
        for sentence_key in sentence_keys:
            template = self.sentences[sentence_key]
            split = "TRAIN" if sentence_key in train_ids else "TEST"
            for label, level_keys in groups.items():
                label_sign = -1 if label == "negative" else 1
                for level_key in level_keys:
                    words = self.words[level_key]
                    for word_index, word in enumerate(words, start=1):
                        filled = template.format(filler=word)
                        rows.append(
                            {
                                "sentence_id": sentence_key,
                                "level_id": level_key,
                                "word_id": str(word_index),
                                "word": word,
                                "sentence": filled,
                                "label": label,
                                "label_sign": label_sign,
                                "split": split,
                            }
                        )
        return pd.DataFrame(rows)

    def get_templated_sentences(
        self,
        train_split=0.8,
        template=None,
        seed=None,
    ) -> pd.DataFrame:

        # TODO: Potentially add a cache here in case we call it a couple of times that way it won't recompute it every time

        if seed is None:
            seed = self.seed
        
        df = self.get_labeled_sentences(train_split=train_split, seed=seed)
        df = df.copy()
        template = template or self.template
        if not template:
            raise ValueError("template must be provided to format sentences")
        df["prompt"] = [
            template.format(sentence=sentence) for sentence in df["sentence"]
        ]
        return df

def main():
    config = Config("config/test.yaml")
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 4,
                        template = SENTIMENT_ABLATION_TEMPLATE, config = config)
    templated_sentences_df = data.get_templated_sentences()
    print(templated_sentences_df)

if __name__ == "__main__":
    main()
