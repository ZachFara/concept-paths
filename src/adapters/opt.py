from __future__ import annotations

from typing import Any, List, Tuple

import torch

from .base import ModelAdapter


class OPTAdapter(ModelAdapter):
    """
    OPT-style decoder blocks (facebook/opt-*).
    Residual: decoder layer output.
    MLP activation: fc1 output (pre-activation).
    """

    adapter_name = "opt"

    def list_layers(self) -> List[Any]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            return list(self.model.model.decoder.layers)
        raise ValueError("OPTAdapter expects model.model.decoder.layers")

    def _capture_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        layers = self.list_layers()
        resid_saves: List[torch.Tensor] = []
        mlp_saves: List[torch.Tensor] = []
        hooks = []

        def resid_hook(_, __, output):
            # OPT decoder layer returns tuple (hidden_states,)
            if isinstance(output, tuple):
                resid_saves.append(output[0].detach())
            else:
                resid_saves.append(output.detach())

        def mlp_hook(_, __, output):
            mlp_saves.append(output.detach())

        for block in layers:
            hooks.append(block.register_forward_hook(resid_hook))
            if hasattr(block, "fc1"):
                hooks.append(block.fc1.register_forward_hook(mlp_hook))
            else:
                hooks.append(block.register_forward_hook(mlp_hook))

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()
        return resid_saves, mlp_saves

    def _capture_batch_with_ablation(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, ablate_layer: int, neuron_idx: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        layers = self.list_layers()
        resid_saves: List[torch.Tensor] = []
        mlp_saves: List[torch.Tensor] = []
        hooks = []

        def resid_hook(_, __, output):
            if isinstance(output, tuple):
                resid_saves.append(output[0].detach())
            else:
                resid_saves.append(output.detach())

        def mlp_hook(layer_idx: int):
            def _hook(_, __, output):
                out = output[0] if isinstance(output, tuple) else output
                if layer_idx == ablate_layer:
                    out[:, :, neuron_idx] = 0
                mlp_saves.append(out.detach())
                return out

            return _hook

        for idx, block in enumerate(layers):
            hooks.append(block.register_forward_hook(resid_hook))
            if hasattr(block, "fc1"):
                hooks.append(block.fc1.register_forward_hook(mlp_hook(idx)))
            else:
                hooks.append(block.register_forward_hook(mlp_hook(idx)))

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()
        return resid_saves, mlp_saves
