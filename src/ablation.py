"""
1. Get data
2. Template data
3. Get the probability of ' positive' vs. ' negative'
4. Identify directions from previous results
5. Ablate neurons identified from directions
6. Perform linear probing ablation
"""

import math
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

    def get_labeled_sentences(self, train_split=0.8, split_by="sentence_id", seed=0):
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

class GPT2Ablator:
    def __init__(self,
                 data:AblationData,
                 template:str,
                 top_k:int,
                 positive_string:str = " positive",
                 negative_string:str = " negative"
                 ):
        self.model = GPT2(n = 10)
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

    def get_templated_sentences(self, train_split = 0.8) -> pd.DataFrame:
        df = self.data.get_labeled_sentences(train_split = train_split)
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

    def get_residual_activations(self, df):

        df = df.copy()
        
        df_with_resids = self.model.add_x_residuals_to_df(df, sentence_column = "prompt")

        return df_with_resids

    def train_linear_probes(self, df, epochs=200, lr=1e-2, seed=0):

        df = df.copy()
        df_with_resids = self.get_residual_activations(df)

        self.linear_probes = {}
        for layer_id in sorted(df_with_resids["layer"].unique()):
            layer_df = df_with_resids[df_with_resids["layer"] == layer_id]
            train_df = layer_df[layer_df["split"] == "TRAIN"]
            test_df = layer_df[layer_df["split"] == "TEST"]

            X_train = torch.stack(train_df["hidden_last"].tolist())
            y_train = torch.tensor(
                (train_df["label"] == "positive").astype(int).values,
                dtype=torch.float32,
            ).unsqueeze(1)
            X_test = torch.stack(test_df["hidden_last"].tolist())
            y_test = torch.tensor(
                (test_df["label"] == "positive").astype(int).values,
                dtype=torch.float32,
            ).unsqueeze(1)

            torch.manual_seed(seed)
            model = torch.nn.Linear(self.n_neurons, 1)
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.BCEWithLogitsLoss()

            for _ in range(int(epochs)):
                optim.zero_grad()
                logits = model(X_train)
                loss = loss_fn(logits, y_train)
                loss.backward()
                optim.step()

            with torch.no_grad():
                train_logits = model(X_train).squeeze(1)
                test_logits = model(X_test).squeeze(1)
                train_pred = (torch.sigmoid(train_logits) >= 0.5).float()
                test_pred = (torch.sigmoid(test_logits) >= 0.5).float()
                train_acc = (train_pred == y_train.squeeze(1)).float().mean().item()
                test_acc = (test_pred == y_test.squeeze(1)).float().mean().item()

            self.linear_probes[layer_id] = {
                "weight": model.weight.detach().cpu().squeeze(0),
                "bias": model.bias.detach().cpu().squeeze(0),
                "train_acc": train_acc,
                "test_acc": test_acc,
            }

        return self.linear_probes

    def linear_probe_logits(self, prompt, top_k=None):
        if not self.linear_probes:
            raise RuntimeError("Linear probes not trained. Call train_linear_probes first.")
        if top_k is None:
            top_k = self.top_k

        ablation_map = {}
        for layer_id, probe in self.linear_probes.items():
            weights = probe["weight"]
            _, indices = torch.topk(torch.abs(weights), int(top_k))
            ablation_map[layer_id] = indices.tolist()

        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        with self.llm.trace(tokenized):
            for layer_id, neuron_ids in ablation_map.items():
                self.llm.transformer.h[layer_id].output[0][:, :, neuron_ids] = 0
            logits = self.llm.lm_head.output[0].save()
        return logits.detach().cpu()

    def normal_probas(self, prompt):
        logits = self.baseline_logits(prompt)
        return self.get_probas(logits)

    def random_probas(self, prompt, seed=0):
        logits = self.random_ablation(prompt, seed=seed)
        return self.get_probas(logits)

    def linear_probas(self, prompt, top_k=None):
        logits = self.linear_probe_logits(prompt, top_k=top_k)
        return self.get_probas(logits)
    
    def fill_test_df(self, df, seed=0, top_k=None):
        df = df.copy()
        test_df = df[df["split"] == "TEST"].copy()

        normal_probs = []
        random_probs = []
        linear_probs = []

        for _, row in test_df.iterrows():
            prompt = row["prompt"]
            normal_probs.append(self.normal_probas(prompt))
            random_probs.append(self.random_probas(prompt, seed=seed))
            linear_probs.append(self.linear_probas(prompt, top_k=top_k))

        test_df[["normal_pos", "normal_neg"]] = pd.DataFrame(
            normal_probs, index=test_df.index
        )
        test_df[["random_pos", "random_neg"]] = pd.DataFrame(
            random_probs, index=test_df.index
        )
        test_df[["linear_pos", "linear_neg"]] = pd.DataFrame(
            linear_probs, index=test_df.index
        )

        return test_df

    def score_probas_df(self, scored_df):
        df = scored_df.copy()
        results = []
        for method in ["normal", "random", "linear"]:
            pos_col = f"{method}_pos"
            neg_col = f"{method}_neg"
            is_pos = df["label"] == "positive"
            prob = df[pos_col].where(is_pos, df[neg_col])
            nll = (-torch.log(torch.tensor(prob.values))).mean().item()
            pred_pos = df[pos_col] >= df[neg_col]
            acc = (pred_pos == is_pos).mean()
            results.append(
                {
                    "method": method,
                    "logit_loss": nll,
                    "accuracy": acc,
                }
            )
        return pd.DataFrame(results)

    def score_significance(
        self,
        scored_df,
        method_a="linear",
        method_b="random",
        n_boot=1000,
        n_perm=1000,
        seed=0,
        eps=1e-9,
    ):
        df = scored_df.copy()
        a_pos = df[f"{method_a}_pos"].values
        a_neg = df[f"{method_a}_neg"].values
        b_pos = df[f"{method_b}_pos"].values
        b_neg = df[f"{method_b}_neg"].values
        is_pos = (df["label"] == "positive").values

        a_prob = torch.tensor(
            [p if pos else n for p, n, pos in zip(a_pos, a_neg, is_pos)],
            dtype=torch.float32,
        ).clamp(min=eps)
        b_prob = torch.tensor(
            [p if pos else n for p, n, pos in zip(b_pos, b_neg, is_pos)],
            dtype=torch.float32,
        ).clamp(min=eps)

        delta = (-torch.log(b_prob)) - (-torch.log(a_prob))
        obs_mean = delta.mean().item()

        rng = torch.Generator().manual_seed(int(seed))
        boot_means = []
        n = len(delta)
        for _ in range(int(n_boot)):
            idx = torch.randint(0, n, (n,), generator=rng)
            boot_means.append(delta[idx].mean().item())
        boot_means = sorted(boot_means)
        lo = boot_means[int(0.025 * len(boot_means))]
        hi = boot_means[int(0.975 * len(boot_means))]

        perm_means = []
        for _ in range(int(n_perm)):
            signs = torch.randint(0, 2, (n,), generator=rng) * 2 - 1
            perm_means.append((delta * signs).mean().item())
        perm_means = torch.tensor(perm_means)
        perm_p = (perm_means.abs() >= abs(obs_mean)).float().mean().item()

        a_correct = (a_pos >= a_neg) == is_pos
        b_correct = (b_pos >= b_neg) == is_pos
        wins_a = int(((a_correct == True) & (b_correct == False)).sum())
        wins_b = int(((a_correct == False) & (b_correct == True)).sum())
        n_wins = wins_a + wins_b

        def binom_cdf(k, n, p):
            return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))

        if n_wins == 0:
            sign_p = 1.0
        else:
            k = min(wins_a, wins_b)
            sign_p = 2 * binom_cdf(k, n_wins, 0.5)
            sign_p = min(1.0, sign_p)

        return pd.DataFrame(
            [
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_nll_gap": obs_mean,
                    "boot_ci_low": lo,
                    "boot_ci_high": hi,
                    "perm_p_value": perm_p,
                    "sign_test_p_value": sign_p,
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                }
            ]
        )

    

    
def rand_norm_linear_single_prompt():
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 3)
    ablator = GPT2Ablator(data, SENTIMENT_ABLATION_TEMPLATE, 100)
    templated_sentences_df = ablator.get_templated_sentences(train_split = .5)

    test_prompt = templated_sentences_df['prompt'].loc[0]

    print(f"Utilizing Prompt: {test_prompt}")

    random_ablated_logits = ablator.random_ablation(test_prompt)
    normal_logits = ablator.baseline_logits(test_prompt)

    normal_prob_pos, normal_prob_neg = ablator.get_probas(normal_logits)
    random_prob_pos, random_prob_neg = ablator.get_probas(random_ablated_logits)


    ablator.train_linear_probes(templated_sentences_df, epochs=100)
    linear_logits = ablator.linear_probe_logits(test_prompt)

    linear_prob_pos, linear_prob_neg = ablator.get_probas(linear_logits)
    
    print(f"Normal p(pos)={normal_prob_pos:.4f} p(neg)={normal_prob_neg:.4f}")
    print(f"Ablated p(pos)={random_prob_pos:.4f} p(neg)={random_prob_neg:.4f}")
    print(f"Linear probe p(pos)={linear_prob_pos:.4f} p(neg)={linear_prob_neg:.4f}")
    for layer_id, stats in sorted(ablator.linear_probes.items()):
        print(
            f"Layer {layer_id} probe acc: train={stats['train_acc']:.3f} "
            f"test={stats['test_acc']:.3f}"
        )

def score_df():
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 4)
    ablator = GPT2Ablator(data, SENTIMENT_ABLATION_TEMPLATE, 10)
    templated_sentences_df = ablator.get_templated_sentences(train_split = .8)
    ablator.train_linear_probes(templated_sentences_df, epochs=100)
    scored_df = ablator.fill_test_df(templated_sentences_df)
    scored_summary = ablator.score_probas_df(scored_df)
    scored_significance = ablator.score_significance(scored_df)
    print(scored_summary)
    print(scored_significance)
    scored_summary.to_csv("summary.csv")
    scored_significance.to_csv("significance.csv")

def main():
    # rand_norm_linear_single_prompt()
    score_df()

if __name__ == "__main__":
    main()
