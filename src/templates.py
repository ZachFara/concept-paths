
import pandas as pd

SENTIMENT_SENTENCES = {
        "sentence.1": "The overall experience was {filler}",
        "sentence.2": "The outcome felt {filler}",
        "sentence.3": "The result was {filler}",
        "sentence.4": "The quality of the work was {filler}",
        "sentence.5": "The performance was {filler}",
        "sentence.6": "The system behavior seemed {filler}",
        "sentence.7": "The response was {filler}",
        "sentence.8": "The evaluation was {filler}",
        "sentence.9": "The overall impression was {filler}",
        "sentence.10": "The final assessment was {filler}",
}

SENTIMENT_WORDS = {
        "level.1": ["abysmal", "atrocious", "awful"],
        "level.2": ["terrible", "horrible", "dreadful"],
        "level.3": ["bad", "poor", "weak"],
        "level.4": ["mediocre", "uninspired", "lackluster"],
        "level.5": ["okay", "fine", "average"],
        "level.6": ["decent", "solid", "pleasant"],
        "level.7": ["good", "enjoyable", "satisfying"],
        "level.8": ["great", "impressive", "strong"],
        "level.9": ["excellent", "fantastic", "outstanding"],
        "level.10": ["exceptional", "remarkable", "superb"],
}

NULL_WORDS = {
        "level.1": ["standard", "typical", "usual"],
        "level.2": ["routine", "regular", "normal"],
        "level.3": ["conventional", "common", "customary"],
        "level.4": ["generic", "general", "baseline"],
        "level.5": ["ordinary", "commonplace", "run-of-the-mill"],
        "level.6": ["expected", "predictable", "familiar"],
        "level.7": ["neutral", "unremarkable", "matter-of-fact"],
        "level.8": ["consistent", "stable", "steady"],
        "level.9": ["typical", "characteristic", "representative"],
        "level.10": ["standardized", "regularized", "conventionalized"],
}

RUDENESS_SENTENCES = {
        "sentence.1": "The response was {filler}.",
        "sentence.2": "The reply felt {filler}.",
        "sentence.3": "The tone was {filler}.",
        "sentence.4": "The message came across as {filler}.",
        "sentence.5": "The wording seemed {filler}.",
        "sentence.6": "The phrasing was {filler}.",
        "sentence.7": "The comment sounded {filler}.",
        "sentence.8": "The remark was {filler}.",
        "sentence.9": "The note was {filler}.",
        "sentence.10": "The statement felt {filler}.",
}

RUDENESS_WORDS = {
        "level.1": ["hostile", "abrasive", "harsh"],
        "level.2": ["rude", "snide", "dismissive"],
        "level.3": ["blunt", "curt", "short"],
        "level.4": ["brusque", "sharp", "cold"],
        "level.5": ["neutral", "plain", "matter-of-fact"],
        "level.6": ["civil", "courteous", "respectful"],
        "level.7": ["polite", "considerate", "kind"],
        "level.8": ["gracious", "warm", "friendly"],
        "level.9": ["thoughtful", "gentle", "tactful"],
        "level.10": ["deferential", "very polite", "exceptionally kind"],
}

FORMALITY_SENTENCES = {
        "sentence.1": "The response was {filler}.",
        "sentence.2": "The reply felt {filler}.",
        "sentence.3": "The tone was {filler}.",
        "sentence.4": "The message came across as {filler}.",
        "sentence.5": "The wording seemed {filler}.",
        "sentence.6": "The phrasing was {filler}.",
        "sentence.7": "The comment sounded {filler}.",
        "sentence.8": "The remark was {filler}.",
        "sentence.9": "The note was {filler}.",
        "sentence.10": "The statement felt {filler}.",
}

FORMALITY_WORDS = {
        "level.1": ["casual", "informal", "relaxed"],
        "level.2": ["conversational", "laid-back", "everyday"],
        "level.3": ["plain", "simple", "direct"],
        "level.4": ["neutral", "straightforward", "standard"],
        "level.5": ["measured", "polished", "proper"],
        "level.6": ["formal", "professional", "businesslike"],
        "level.7": ["official", "ceremonial", "authoritative"],
        "level.8": ["prestigious", "dignified", "stately"],
        "level.9": ["elevated", "refined", "sophisticated"],
        "level.10": ["highly formal", "very formal", "extremely formal"],
}

class Template:
    def __init__(self, sentence_dict:dict, words_dict:dict):
        self.sentence_dict = sentence_dict
        self.words_dict = words_dict

    def get_sentence(
            self,
            template_id:str,
            level_id:str,
            sentiment_word_id:str
            ):

        # Assertions
        template_num = int(template_id)
        assert 1 <= template_num <= len(self.sentence_dict)
        complete_template_id = f"sentence.{template_num}"
        assert complete_template_id in self.sentence_dict

        level_num = int(level_id)
        assert 1 <= level_num <= len(self.words_dict)
        complete_level_id = f"level.{level_num}"
        assert complete_level_id in self.words_dict

        word_num = int(sentiment_word_id)
        assert 1 <= word_num <= 3
        complete_word_num = word_num - 1 # For 0 based indexing

        # Get the template that we want to fill
        template =  self.sentence_dict[complete_template_id]

        # Get the sentiment word that we wanna add to it
        word = self.words_dict[complete_level_id][complete_word_num]

        # Fill the template
        filled_template = template.format(filler = word)

        return filled_template

    def get_all_sentences(self):
        rows = []
        sentence_keys = sorted(
            self.sentence_dict.keys(),
            key=lambda k: int(k.split(".")[1]),
        )
        level_keys = sorted(
            self.words_dict.keys(),
            key=lambda k: int(k.split(".")[1]),
        )
        for sentence_key in sentence_keys:
            template = self.sentence_dict[sentence_key]
            for level_key in level_keys:
                words = self.words_dict[level_key]
                for word_index, word in enumerate(words, start=1):
                    filled = template.format(filler=word)
                    rows.append(
                        {
                            "sentence_id": sentence_key,
                            "level_id": level_key,
                            "item_id": sentence_key + level_key,
                            "word_id": str(word_index),
                            "word": word,
                            "sentence": filled,
                        }
                    )
        return pd.DataFrame(rows)

    def get_grid_sentences(
        self,
        sentence_ids=None,
        level_ids=None,
        word_ids=None,
    ):
        sentence_keys = sorted(
            self.sentence_dict.keys(),
            key=lambda k: int(k.split(".")[1]),
        )
        level_keys = sorted(
            self.words_dict.keys(),
            key=lambda k: int(k.split(".")[1]),
        )

        if sentence_ids is not None:
            normalized_sentence_ids = [
                sid if str(sid).startswith("sentence.") else f"sentence.{int(sid)}"
                for sid in sentence_ids
            ]
            sentence_keys = [k for k in sentence_keys if k in normalized_sentence_ids]
        if level_ids is not None:
            normalized_level_ids = [
                lid if str(lid).startswith("level.") else f"level.{int(lid)}"
                for lid in level_ids
            ]
            level_keys = [k for k in level_keys if k in normalized_level_ids]

        word_indices = None
        if word_ids is not None:
            word_indices = [int(wid) - 1 for wid in word_ids]

        rows = []
        for sentence_key in sentence_keys:
            template = self.sentence_dict[sentence_key]
            for level_key in level_keys:
                words = self.words_dict[level_key]
                if word_indices is None:
                    selected = enumerate(words, start=1)
                else:
                    selected = (
                        (i + 1, words[i])
                        for i in word_indices
                        if 0 <= i < len(words)
                    )
                for word_index, word in selected:
                    filled = template.format(filler=word)
                    rows.append(
                        {
                            "sentence_id": sentence_key,
                            "level_id": level_key,
                            "item_id": sentence_key + level_key,
                            "word_id": str(word_index),
                            "word": word,
                            "sentence": filled,
                        }
                    )
        return pd.DataFrame(rows)


def main():
    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    sentence = temp.get_sentence(1, 1, 1)
    print(sentence)

    all_sentences = temp.get_all_sentences()
    print(f"Total sentences: {len(all_sentences)}")
    print("Sample of all sentiment sentences:")
    print(all_sentences.head(10))

    print("Sample of grid sentences working:")
    grid = temp.get_grid_sentences(sentence_ids=[1, 2], level_ids=[1], word_ids=[1, 2])
    print(grid)


if __name__ == "__main__":
    main()
