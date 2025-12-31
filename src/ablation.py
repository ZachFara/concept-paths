import pandas as pd
import torch

from src.capture import GPT2
from src.logs import setup_logger
from src.templates import SENTIMENT_SENTENCES, SENTIMENT_WORDS

logger = setup_logger(__name__)

SENTIMENT_ABLATION_TEMPLATE = "Sentence: {sentence} Sentiment (positive/negative):"
SENTIMENT_ABLATION_TEMPLATE = "Sentence: {sentence} Sentiment (positive/negative):"


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


class GPT2TopKAblator:
    def __init__(self, top_k=50, n_last_tokens=1):
        self.top_k = int(top_k)
        self.n_last_tokens = int(n_last_tokens)
        self.model = GPT2().LLM
        self.tokenizer = self.model.tokenizer
        self.d_model = int(self.model.config.n_embd)

    def _tokenize(self, prompt):
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def _log_prompt_tokens(self, prompt):
        tokenized = self._tokenize(prompt)
        input_ids = tokenized["input_ids"][0].tolist()
        decoded = [self.tokenizer.decode(tid) for tid in input_ids]
        logger.info("Prompt tokens: %s", input_ids)
        logger.info("Prompt decoded: %s", decoded)
        return tokenized

    def _log_target_tokens(self, positive_word, negative_word):
        pos_ids = self.tokenizer.encode(positive_word, add_special_tokens=False)
        neg_ids = self.tokenizer.encode(negative_word, add_special_tokens=False)
        if not pos_ids or not neg_ids:
            raise ValueError("target words must tokenize to at least one token")
        pos_last = pos_ids[-1]
        neg_last = neg_ids[-1]
        logger.info(
            "Target tokens (positive/negative): %s/%s",
            pos_last,
            neg_last,
        )
        logger.info(
            "Target decoded (positive/negative): %s/%s",
            self.tokenizer.decode(pos_last),
            self.tokenizer.decode(neg_last),
        )
        return pos_last, neg_last

    def _get_layer_ids(self, layer_ids):
        if layer_ids is not None:
            return list(layer_ids)
        return list(range(len(self.model.transformer.h)))

    def _normalize_direction(self, direction):
        vec = torch.tensor(direction, dtype=torch.float32)
        if vec.shape != (self.d_model,):
            raise ValueError("direction must be length d_model")
        return vec

    def _get_direction_map(self, direction_by_layer, layer_ids):
        if isinstance(direction_by_layer, dict):
            direction_map = {}
            for layer_id in layer_ids:
                if layer_id not in direction_by_layer:
                    raise ValueError(f"missing direction for layer {layer_id}")
                direction_map[layer_id] = self._normalize_direction(
                    direction_by_layer[layer_id]
                )
            return direction_map
        direction = self._normalize_direction(direction_by_layer)
        return {layer_id: direction for layer_id in layer_ids}

    def select_topk_neurons(
        self,
        prompts,
        direction_by_layer,
        layer_ids=None,
        score_mode="abs",
    ):
        layer_ids = self._get_layer_ids(layer_ids)
        direction_map = self._get_direction_map(direction_by_layer, layer_ids)

        scores = {
            layer_id: torch.zeros(self.d_model, dtype=torch.float32)
            for layer_id in layer_ids
        }

        for prompt in prompts:
            tokenized = self._log_prompt_tokens(prompt)
            input_ids = tokenized["input_ids"]
            last_idx = max(int(input_ids.shape[1]) - 1, 0)

            with self.model.trace(tokenized):
                mlp_outputs = {
                    layer_id: self.model.transformer.h[layer_id].mlp.output[0].save()
                    for layer_id in layer_ids
                }

            for layer_id, mlp_out in mlp_outputs.items():
                vec = mlp_out[0, last_idx, :].detach().cpu()
                direction = direction_map[layer_id]
                contrib = vec * direction
                if score_mode == "abs":
                    contrib = torch.abs(contrib)
                scores[layer_id] += contrib

        topk = {}
        for layer_id in layer_ids:
            values, indices = torch.topk(scores[layer_id], self.top_k)
            topk[layer_id] = indices.tolist()
            logger.info(
                "Ablating layer %s top_k=%s neurons: %s",
                layer_id,
                self.top_k,
                topk[layer_id],
            )
        return topk

    def _forward_logits(self, prompt, ablation_map=None):
        tokenized = self._log_prompt_tokens(prompt)
        with self.model.trace(tokenized):
            if ablation_map:
                for layer_id, neuron_ids in ablation_map.items():
                    mlp_out = self.model.transformer.h[layer_id].mlp.output[0]
                    mlp_out[:, :, neuron_ids] = 0
            logits = self.model.lm_head.output[0].save()
        return logits.detach().cpu()

    def ablate_logits(self, prompts, ablation_map):
        logits_list = []
        for prompt in prompts:
            logits_list.append(self._forward_logits(prompt, ablation_map=ablation_map))
        return logits_list

    def word_to_last_token_id(self, word):
        token_ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            raise ValueError("word tokenization produced no tokens")
        return token_ids[-1]

    def compute_logit_gap(self, logits, positive_words, negative_words):
        """Uses the last-token id for each word; multi-token words are approximated by the last token."""
        pos_ids = [self.word_to_last_token_id(w) for w in positive_words]
        neg_ids = [self.word_to_last_token_id(w) for w in negative_words]
        last_logits = logits[0, -1, :]
        pos_mean = last_logits[pos_ids].mean().item()
        neg_mean = last_logits[neg_ids].mean().item()
        return pos_mean - neg_mean

    def run_sentiment_ablation(
        self,
        data: AblationData,
        direction_by_layer,
        top_k=None,
        layer_ids=None,
        max_prompts=None,
        positive_word=" positive",
        negative_word=" negative",
    ):
        if top_k is not None:
            self.top_k = int(top_k)
        df = data.get_labeled_sentences()
        sentences = df["sentence"].unique().tolist()
        if max_prompts is not None:
            sentences = sentences[: int(max_prompts)]
        prompts = [
            SENTIMENT_ABLATION_TEMPLATE.format(sentence=sentence) for sentence in sentences
        ]

        pos_id, neg_id = self._log_target_tokens(positive_word, negative_word)
        _ = (pos_id, neg_id)

        ablation_map = self.select_topk_neurons(
            prompts=prompts,
            direction_by_layer=direction_by_layer,
            layer_ids=layer_ids,
        )

        baseline_logits = [self._forward_logits(p) for p in prompts]
        ablated_logits = self.ablate_logits(prompts, ablation_map)

        rows = []
        for prompt, base, ablated in zip(prompts, baseline_logits, ablated_logits):
            base_gap = self.compute_logit_gap(
                base, [positive_word], [negative_word]
            )
            ablated_gap = self.compute_logit_gap(
                ablated, [positive_word], [negative_word]
            )
            rows.append(
                {
                    "prompt": prompt,
                    "baseline_logit_gap": base_gap,
                    "ablated_logit_gap": ablated_gap,
                    "logit_gap_drop": base_gap - ablated_gap,
                }
            )
        return pd.DataFrame(rows)


def main():
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 3)
    abl = GPT2TopKAblator(top_k=50)

    direction = torch.randn(abl.d_model)
    results = abl.run_sentiment_ablation(
        data=data,
        direction_by_layer=direction,
        max_prompts=3,
    )
    print(results)


if __name__ == "__main__":
    main()
