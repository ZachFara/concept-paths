"""
1. Get data
2. Template data
3. Get the probability of ' positive' vs. ' negative'
4. Identify directions from previous results
5. Ablate neurons identified from directions
6. Perform linear probing ablation
"""

import pandas as pd
import torch

from src.capture import GPT2
from src.logs import setup_logger
from src.templates import SENTIMENT_SENTENCES, SENTIMENT_WORDS

logger = setup_logger(__name__)

SENTIMENT_ABLATION_TEMPLATE = "Sentence: {sentence}. Sentiment (positive/negative):"

class AblationData:
    def __init__(self, sentences, words, cutoff):
        """Cutoff is the number of bottom levels treated as negative and top levels treated as positive."""
        self.sentences = sentences
        self.words = words
        self.cutoff = int(cutoff)
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

    def get_labeled_sentences(self):
        groups = self.get_level_groups()
        rows = []
        for sentence_key in self._sorted_sentence_keys():
            template = self.sentences[sentence_key]
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
                            }
                        )
        return pd.DataFrame(rows)

class GPT2Ablator:
    def __init__(self,
                 data:AblationData,
                 template:str,
                 top_k:int,
                 positive_string:str = " positive",
                 negative_string:str = " negative"
                 ):
        self.model = GPT2()
        self.llm = self.model.LLM
        self.tokenizer = self.llm.tokenizer
        self.n_layers = self.llm.config.n_layer
        self.n_neurons = self.llm.config.n_embd
        self.data = data
        assert isinstance(data, AblationData)
        self.template = template
        self.top_k = top_k
        self.positive_string = positive_string
        self.negative_string = negative_string
        self.pos_id, self.neg_id = self.get_pos_neg_ids()

        print(f"This ablator will ablate: {(self.top_k / self.n_neurons) * 100:.2f}% of the neurons in this model")

    def get_templated_sentences(self) -> pd.DataFrame:
        df = self.data.get_labeled_sentences()
        df = df.copy()
        df["prompt"] = [
            self.template.format(sentence=sentence) for sentence in df["sentence"]
        ]
        return df

    def random_ablation(self, prompt, seed = 0):
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        with self.llm.trace(tokenized):
            for layer in range(self.n_layers):
                random_neuron_indices = torch.randperm(
                    self.n_neurons,
                    generator=torch.Generator().manual_seed(seed + layer),
                )[: self.top_k]
                self.llm.transformer.h[layer].output[0][:, :, random_neuron_indices] = 0
            logits = self.llm.lm_head.output[0].save()
        return logits.detach().cpu()

    def baseline_logits(self, prompt):
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        with self.llm.trace(tokenized):
            logits = self.llm.lm_head.output[0].save()
        return logits.detach().cpu()

    def get_pos_neg_ids(self):
        pos_ids = self.tokenizer.encode(self.positive_string, add_special_tokens=False)
        neg_ids = self.tokenizer.encode(self.negative_string, add_special_tokens=False)
        if len(pos_ids) != 1:
            logger.warning(
                "Positive string tokenized into %s tokens: %s",
                len(pos_ids),
                pos_ids,
            )
        if len(neg_ids) != 1:
            logger.warning(
                "Negative string tokenized into %s tokens: %s",
                len(neg_ids),
                neg_ids,
            )
        if not pos_ids or not neg_ids:
            raise ValueError("Positive or negative string tokenized to empty list")
        return pos_ids[-1], neg_ids[-1]

    def get_probas(self, logits):
        last_logits = logits[-1, [self.pos_id, self.neg_id]] # Assume logits has two dimensions
        probs = torch.softmax(last_logits, dim=-1)
        return probs.tolist()

def main():
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 3)
    ablator = GPT2Ablator(data, SENTIMENT_ABLATION_TEMPLATE, 100)
    templated_sentences_df = ablator.get_templated_sentences()

    test_prompt = templated_sentences_df['prompt'].loc[0]

    print(f"Utilizing Prompt: {test_prompt}")

    random_ablated_logits = ablator.random_ablation(test_prompt)
    normal_logits = ablator.baseline_logits(test_prompt)

    normal_prob_pos, normal_prob_neg = ablator.get_probas(normal_logits)
    random_prob_pos, random_prob_neg = ablator.get_probas(random_ablated_logits)

    print(f"Normal p(pos)={normal_prob_pos:.4f} p(neg)={normal_prob_neg:.4f}")
    print(f"Ablated p(pos)={random_prob_pos:.4f} p(neg)={random_prob_neg:.4f}")


if __name__ == "__main__":
    main()
