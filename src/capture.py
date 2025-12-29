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
            resid_last = model.transformer.h[-1].output[0].save()

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
        out_df["residuals"] = residuals
        out_df['layer'] = len(model.transformer.h) - 1 # The last layer

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
        layer_ids = list(range(len(model.transformer.h)))


def main():

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

if __name__ == "__main__":
    main()
