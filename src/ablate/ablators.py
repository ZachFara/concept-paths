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

logger = setup_logger(__name__)

class GPT2Ablator:
    def __init__(self,
                 data:AblationData,
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
        self.top_k = top_k
        self.positive_string = positive_string
        self.negative_string = negative_string
        self.pos_id, self.neg_id = self.get_pos_neg_ids()
        self._pca_ablation_cache = {}

        logger.info(f"This ablator will ablate: {(self.top_k / self.n_neurons) * 100:.2f}% of the neurons in this model")

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

    def get_pca_ablation_map(self, pca_dict, top_k=None, train_split=0.8):
        if top_k is None:
            top_k = self.top_k
        cache_key = (id(pca_dict), int(top_k), float(train_split))
        if cache_key in self._pca_ablation_cache:
            return self._pca_ablation_cache[cache_key]
        df = self.data.get_templated_sentences(train_split=train_split)
        ablation_map = self.get_directions(df, pca_dict)
        self._pca_ablation_cache[cache_key] = ablation_map
        return ablation_map

    def pca_ablation_logits(self, prompt, pca_dict, top_k=None):
        if top_k is None:
            top_k = self.top_k
        ablation_map = self.get_pca_ablation_map(pca_dict, top_k=top_k)

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

    def get_directions(self, df, pca_dict):

        train_df = df[df['split'] == "TRAIN"]
        df_with_resids = self.get_residual_activations(train_df)
        ablation_map = {}

        for layer_id, pca in pca_dict.items():
            layer_df = df_with_resids[df_with_resids["layer"] == layer_id]
            if layer_df.empty:
                raise ValueError(f"No residuals found for layer {layer_id}")

            if not hasattr(pca, "components_"):
                raise ValueError(f"PCA object for layer {layer_id} missing components_")
            direction_vec = torch.tensor(pca.components_[0], dtype=torch.float32)
            if direction_vec.shape != (self.n_neurons,):
                raise ValueError(
                    f"PCA component for layer {layer_id} has shape {direction_vec.shape}, "
                    f"expected ({self.n_neurons},)"
                )

            hidden_stack = torch.stack(layer_df["hidden_last"].tolist())
            scores = torch.mean(torch.abs(hidden_stack * direction_vec), dim=0)
            _, indices = torch.topk(scores, int(self.top_k))
            ablation_map[layer_id] = indices.tolist()

        return ablation_map

    def normal_probas(self, prompt):
        logits = self.baseline_logits(prompt)
        return self.get_probas(logits)

    def random_probas(self, prompt, seed=0):
        logits = self.random_ablation(prompt, seed=seed)
        return self.get_probas(logits)

    def linear_probas(self, prompt, top_k=None):
        logits = self.linear_probe_logits(prompt, top_k=top_k)
        return self.get_probas(logits)

    def pca_probas(self, prompt, pca_dict, top_k=None):
        logits = self.pca_ablation_logits(prompt, pca_dict, top_k=top_k)
        return self.get_probas(logits)

    def fill_test_df(self, df, pca_dict, seed=0, top_k=None):
        df = df.copy()
        test_df = df[df["split"] == "TEST"].copy()

        normal_probs = []
        random_probs = []
        linear_probs = []
        pca_probs = []

        for _, row in test_df.iterrows():
            prompt = row["prompt"]
            normal_probs.append(self.normal_probas(prompt))
            random_probs.append(self.random_probas(prompt, seed=seed))
            linear_probs.append(self.linear_probas(prompt, top_k=top_k))
            pca_probs.append(self.pca_probas(prompt, pca_dict, top_k=top_k))

        test_df[["normal_pos", "normal_neg"]] = pd.DataFrame(
            normal_probs, index=test_df.index
        )
        test_df[["random_pos", "random_neg"]] = pd.DataFrame(
            random_probs, index=test_df.index
        )
        test_df[["linear_pos", "linear_neg"]] = pd.DataFrame(
            linear_probs, index=test_df.index
        )
        test_df[["pca_pos", "pca_neg"]] = pd.DataFrame(
            pca_probs, index=test_df.index
        )

        return test_df

    def score_probas_df(self, scored_df):
        df = scored_df.copy()
        results = []
        for method in ["normal", "random", "linear", "pca"]:
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

def main():
    data = AblationData(SENTIMENT_SENTENCES, SENTIMENT_WORDS, 4, SENTIMENT_ABLATION_TEMPLATE)
    ablator = GPT2Ablator(data, 10)
    templated_sentences_df = data.get_templated_sentences(train_split = .8)
    
    ablator.train_linear_probes(templated_sentences_df, epochs=100)
    
    # All of this code is just to basically get the deltas to get the PCA objects
    temp = Template(SENTIMENT_SENTENCES, SENTIMENT_WORDS)
    gpt = GPT2()
    df = temp.get_all_sentences()
    df = gpt.add_x_residuals_to_df(df = df, x = None)
    delta = Deltas(df)
    group_cols = ["sentence_id", "layer"]
    mu = delta.compute_mu(group_cols=group_cols)
    deltas_adj = delta.compute_adjacent_deltas(mu, group_cols)
    pca = PCA(deltas_adj)
    pca_dict = pca.get_all_layer_pca(deltas_adj)
    print("Getting scored data from all ablation methods...")
    scored_df = ablator.fill_test_df(templated_sentences_df, pca_dict)
    print(f"Done. Here is the scored data: {scored_df}")
    scored_df.to_csv("scored_data.csv")

if __name__ == "__main__":
    main()
