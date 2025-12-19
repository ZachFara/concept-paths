from __future__ import annotations

from typing import Any, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from .base import ModelAdapter
from ..utils import get_device


class DistilBERTAdapter(ModelAdapter):
    """
    DistilBERT encoder adapter.
    Residual: transformer block output (hidden states).
    MLP: FFN lin1 pre-activation.
    """

    adapter_name = "distilbert"

    def __init__(self, model_name: str, *, device: str | None = None, local_files_only: bool = True):
        self.model_name = model_name
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.model.eval()

    def list_layers(self) -> List[Any]:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layer"):
            return list(self.model.transformer.layer)
        raise ValueError("DistilBERTAdapter expects model.transformer.layer")

    def _capture_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        layers = self.list_layers()
        resid_saves: List[torch.Tensor] = []
        mlp_saves: List[torch.Tensor] = []
        hooks = []

        def resid_hook(_, __, output):
            out = output[0] if isinstance(output, tuple) else output
            resid_saves.append(out.detach())

        def mlp_hook(_, __, output):
            out = output[0] if isinstance(output, tuple) else output
            mlp_saves.append(out.detach())

        for block in layers:
            hooks.append(block.register_forward_hook(resid_hook))
            if hasattr(block, "ffn") and hasattr(block.ffn, "lin1"):
                hooks.append(block.ffn.lin1.register_forward_hook(mlp_hook))
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
            out = output[0] if isinstance(output, tuple) else output
            resid_saves.append(out.detach())

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
            if hasattr(block, "ffn") and hasattr(block.ffn, "lin1"):
                hooks.append(block.ffn.lin1.register_forward_hook(mlp_hook(idx)))
            else:
                hooks.append(block.register_forward_hook(mlp_hook(idx)))

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()
        return resid_saves, mlp_saves
