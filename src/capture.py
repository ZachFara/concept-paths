from nnsight import LanguageModel
from src.logs import setup_logger
from typing import List
import torch
import torch.nn.functional as F
import pandas as pd

logger = setup_logger(__name__)

class GPT2:
    def __init__(self, n = 3):
        self.n = n
        self.LLM = LanguageModel("gpt2", device_map = "auto")
        pass

    def get_last_residual_stream(self, prompt):
        n = self.n

        model = self.LLM
        tokenizer = model.tokenizer
        tokenized_input = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokenized_input["input_ids"][0]

        last_n_ids = input_ids[-n:].tolist()
        last_untokenized_tokens = [tokenizer.decode(id) for id in last_n_ids]

        logger.info("Last token ids: %s", last_n_ids)
        logger.info("Last decoded tokens: %s", last_untokenized_tokens)

        with model.trace(tokenized_input):
            resid_last = model.transformer.layers[-1].output[0].save()

        return resid_last

    def get_x_residual_stream(self, prompt, x=None):
        n = self.n

        model = self.LLM
        tokenizer = model.tokenizer
        tokenized_input = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokenized_input["input_ids"][0]

        last_n_ids = input_ids[-n:].tolist()
        last_untokenized_tokens = [tokenizer.decode(id) for id in last_n_ids]

        logger.info("Last token ids: %s", last_n_ids)
        logger.info("Last decoded tokens: %s", last_untokenized_tokens)

        num_layers = len(model.transformer.h)
        layer_ids = list(range(num_layers)) if x is None else list(x)

        residuals = {}

        with model.trace(tokenized_input) as tracer:
            for i in layer_ids:
                residuals[i] = model.transformer.h[i].output[0].save()

        return residuals
    
    
    def get_multiple_last_residuals(self, prompts:List[str]):
        """Deprecated"""

        model = self.LLM
        residuals = [None] * len(prompts)
        lengths = [None] * len(prompts)

        for i, prompt in enumerate(prompts):

            resid = self.get_last_residual_stream(prompt)
            residuals[i] = resid
            lengths[i] = resid.shape[1]

        max_len = max(lengths)
        padded = [
            F.pad(t, (0, 0, 0, max_len - t.shape[1]))
            for t in residuals
        ]
        stacked = torch.stack(padded)
        return stacked

    def add_padded_residuals_to_df(
        self,
        df
    ):
        prompts = df["sentence"].tolist()
        residuals = []
        lengths = []
        for prompt in prompts:
            resid = self.get_last_residual_stream(prompt)
            residuals.append(resid)
            lengths.append(resid.shape[1])

        max_len = max(lengths) if lengths else 0
        padded = [
            F.pad(t, (0, 0, 0, max_len - t.shape[1])).detach().cpu()
            for t in residuals
        ]

        out_df = df.copy()
        out_df["padded_residual"] = pd.Series(
            [t for t in padded], index=out_df.index, dtype=object
        )
        out_df["seq_len"] = lengths
        out_df["residuals"] = pd.Series(
            [r.detach().cpu() for r in residuals],
            index=out_df.index,
            dtype=object,
        )
        out_df["layer"] = len(self.LLM.transformer.h) - 1  # The last layer

        def pool_last(row):
            resid = row["padded_residual"]
            idx = max(int(row["seq_len"]) - 1, 0)
            return resid[:, idx, :].squeeze(0)

        out_df["hidden_last"] = out_df.apply(pool_last, axis=1)
        return out_df

    def add_x_residuals_to_df(
        self,
        df,
        x = None
    ):
        prompts = df["sentence"].tolist()
        model = self.LLM
        num_layers = len(model.transformer.h)
        layer_ids = list(range(num_layers)) if x is None else list(x)

        per_prompt = []
        for prompt in prompts:
            layer_resids = self.get_x_residual_stream(prompt, x=layer_ids)
            per_prompt.append(layer_resids)

        max_len_by_layer = {}
        for layer_id in layer_ids:
            max_len_by_layer[layer_id] = max(
                layer_resids[layer_id].shape[1] for layer_resids in per_prompt
            ) if per_prompt else 0

        rows = []
        for row, layer_resids in zip(df.to_dict("records"), per_prompt):
            for layer_id in layer_ids:
                resid = layer_resids[layer_id]
                seq_len = resid.shape[1]
                max_len = max_len_by_layer[layer_id]
                padded = F.pad(
                    resid,
                    (0, 0, 0, max_len - seq_len),
                ).detach().cpu()
                hidden_last = padded[:, max(seq_len - 1, 0), :].squeeze(0)
                new_row = dict(row)
                new_row.update(
                    {
                        "padded_residual": padded,
                        "seq_len": seq_len,
                        "residuals": resid.detach().cpu(),
                        "layer": layer_id,
                        "hidden_last": hidden_last,
                    }
                )
                rows.append(new_row)

        return pd.DataFrame(rows)


class Pythia:
    def __init__(self, n=3, model_name="EleutherAI/pythia-160m"):
        self.n = n
        self.model_name = model_name
        self.LLM = LanguageModel(self.model_name, device_map="auto")

    def get_last_residual_stream(self, prompt):
        n = self.n

        model = self.LLM
        tokenizer = model.tokenizer
        tokenized_input = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokenized_input["input_ids"][0]

        last_n_ids = input_ids[-n:].tolist()
        last_untokenized_tokens = [tokenizer.decode(id) for id in last_n_ids]

        logger.info("Last token ids: %s", last_n_ids)
        logger.info("Last decoded tokens: %s", last_untokenized_tokens)

        with model.trace(tokenized_input):
            resid_last = model.gpt_neox.layers[-1].output[0].save()

        return resid_last

    def get_x_residual_stream(self, prompt, x=None):
        n = self.n

        model = self.LLM
        tokenizer = model.tokenizer
        tokenized_input = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokenized_input["input_ids"][0]

        last_n_ids = input_ids[-n:].tolist()
        last_untokenized_tokens = [tokenizer.decode(id) for id in last_n_ids]

        logger.info("Last token ids: %s", last_n_ids)
        logger.info("Last decoded tokens: %s", last_untokenized_tokens)

        num_layers = len(model.gpt_neox.layers)
        layer_ids = list(range(num_layers)) if x is None else list(x)

        residuals = {}

        with model.trace(tokenized_input) as tracer:
            for i in layer_ids:
                residuals[i] = model.gpt_neox.layers[i].output[0].save()

        return residuals

    def add_padded_residuals_to_df(self, df):
        prompts = df["sentence"].tolist()
        residuals = []
        lengths = []
        for prompt in prompts:
            resid = self.get_last_residual_stream(prompt)
            residuals.append(resid)
            lengths.append(resid.shape[1])

        max_len = max(lengths) if lengths else 0
        padded = [
            F.pad(t, (0, 0, 0, max_len - t.shape[1])).detach().cpu()
            for t in residuals
        ]

        out_df = df.copy()
        out_df["padded_residual"] = pd.Series(
            [t for t in padded], index=out_df.index, dtype=object
        )
        out_df["seq_len"] = lengths
        out_df["residuals"] = pd.Series(
            [r.detach().cpu() for r in residuals],
            index=out_df.index,
            dtype=object,
        )
        out_df["layer"] = len(self.LLM.gpt_neox.layers) - 1  # The last layer

        def pool_last(row):
            resid = row["padded_residual"]
            idx = max(int(row["seq_len"]) - 1, 0)
            return resid[:, idx, :].squeeze(0)

        out_df["hidden_last"] = out_df.apply(pool_last, axis=1)
        return out_df

    def add_x_residuals_to_df(self, df, x=None):
        prompts = df["sentence"].tolist()
        model = self.LLM
        num_layers = len(model.gpt_neox.layers)
        layer_ids = list(range(num_layers)) if x is None else list(x)

        per_prompt = []
        for prompt in prompts:
            layer_resids = self.get_x_residual_stream(prompt, x=layer_ids)
            per_prompt.append(layer_resids)

        max_len_by_layer = {}
        for layer_id in layer_ids:
            max_len_by_layer[layer_id] = max(
                layer_resids[layer_id].shape[1] for layer_resids in per_prompt
            ) if per_prompt else 0

        rows = []
        for row, layer_resids in zip(df.to_dict("records"), per_prompt):
            for layer_id in layer_ids:
                resid = layer_resids[layer_id]
                seq_len = resid.shape[1]
                max_len = max_len_by_layer[layer_id]
                padded = F.pad(
                    resid,
                    (0, 0, 0, max_len - seq_len),
                ).detach().cpu()
                hidden_last = padded[:, max(seq_len - 1, 0), :].squeeze(0)
                new_row = dict(row)
                new_row.update(
                    {
                        "padded_residual": padded,
                        "seq_len": seq_len,
                        "residuals": resid.detach().cpu(),
                        "layer": layer_id,
                        "hidden_last": hidden_last,
                    }
                )
                rows.append(new_row)

        return pd.DataFrame(rows)

def basic_test():
    gpt = GPT2()

    residuals = gpt.get_last_residual_stream("This is a test")

    print(f"Residuals (Shape: {residuals.shape}):")
    print(residuals)

    multiple_prompts = [
            "This is a test",
            "This is a longer test"
            ]

    multiple_residuals = gpt.get_multiple_last_residuals(multiple_prompts) 
    print(f"Multiple Residuals (Shape: {multiple_residuals.shape}):")
    print(multiple_residuals)

def testing_x_residuals():
    gpt = GPT2()
    num_layers = len(gpt.LLM.transformer.h)
    layer_ids = [0, num_layers - 1]

    df = pd.DataFrame(
        [
            {
                "sentence_id": "sentence.1",
                "level_id": "level.1",
                "word_id": "1",
                "word": "test",
                "sentence": "This is a test",
            },
            {
                "sentence_id": "sentence.1",
                "level_id": "level.2",
                "word_id": "2",
                "word": "longer",
                "sentence": "This is a longer test",
            },
        ]
    )

    out_df = gpt.add_x_residuals_to_df(df, x=layer_ids)

    expected_cols = {
        "sentence_id",
        "level_id",
        "word_id",
        "word",
        "sentence",
        "padded_residual",
        "seq_len",
        "residuals",
        "layer",
        "hidden_last",
    }
    assert expected_cols.issubset(out_df.columns)
    assert len(out_df) == len(df) * len(layer_ids)
    print(out_df.head(2))


def testing_x_residuals_pythia():
    model = Pythia()
    num_layers = len(model.LLM.gpt_neox.layers)
    layer_ids = [0, num_layers - 1]

    df = pd.DataFrame(
        [
            {
                "sentence_id": "sentence.1",
                "level_id": "level.1",
                "word_id": "1",
                "word": "test",
                "sentence": "This is a test",
            },
            {
                "sentence_id": "sentence.1",
                "level_id": "level.2",
                "word_id": "2",
                "word": "longer",
                "sentence": "This is a longer test",
            },
        ]
    )

    out_df = model.add_x_residuals_to_df(df, x=layer_ids)

    expected_cols = {
        "sentence_id",
        "level_id",
        "word_id",
        "word",
        "sentence",
        "padded_residual",
        "seq_len",
        "residuals",
        "layer",
        "hidden_last",
    }
    assert expected_cols.issubset(out_df.columns)
    assert len(out_df) == len(df) * len(layer_ids)
    print(out_df.head(2))

def main():
    # basic_test()
    # testing_x_residuals()
    testing_x_residuals_pythia()


if __name__ == "__main__":
    main()
