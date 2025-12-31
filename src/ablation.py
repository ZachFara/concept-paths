import pandas as pd
from src.templates import SENTIMENT_SENTENCES, SENTIMENT_WORDS


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




def main():

    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 3)
    ablation_sentences_df = data.get_labeled_sentences() 
    print(ablation_sentences_df)



if __name__ == "__main__":
    main()
