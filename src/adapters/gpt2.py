from __future__ import annotations

from typing import Any, List, Tuple

import torch

from .base import ModelAdapter


class GPT2Adapter(ModelAdapter):
    """
    GPT-2 style blocks: transformer.h list.
    Residual stream: block output (hidden_states).
    MLP activation: block.mlp.c_fc output (pre-activation).
    """

    adapter_name = "gpt2"

    def list_layers(self) -> List[Any]:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        raise ValueError("GPT2Adapter expects model.transformer.h")

    def _capture_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        layers = self.list_layers()
        resid_saves: List[torch.Tensor] = []
        mlp_saves: List[torch.Tensor] = []
        hooks = []

        def resid_hook(_, __, output):
            # output may be tuple (hidden_states, ...) or Tensor
            if isinstance(output, tuple):
                output = output[0]
            resid_saves.append(output.detach())

        def mlp_hook(_, __, output):
            # output: pre-activation linear output [B, S, d_mlp]
            if isinstance(output, tuple):
                output = output[0]
            mlp_saves.append(output.detach())

        for block in layers:
            hooks.append(block.register_forward_hook(resid_hook))
            if hasattr(block, "mlp") and hasattr(block.mlp, "c_fc"):
                hooks.append(block.mlp.c_fc.register_forward_hook(mlp_hook))
            else:
                hooks.append(block.mlp.register_forward_hook(mlp_hook))  # type: ignore[attr-defined]

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()
        return resid_saves, mlp_saves
