from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from ..utils import get_device


@dataclass
class CaptureResult:
    residual: np.ndarray  # [N, L, hidden]
    mlp: np.ndarray  # [N, L, d_mlp]
    metadata: Dict[str, Any]


class ModelAdapter:
    """
    Base adapter for transformer decoder models.
    """

    adapter_name: str = "base"

    def __init__(self, model_name: str, *, device: str | None = None, local_files_only: bool = True):
        self.model_name = model_name
        self.device = get_device(device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

    def list_layers(self) -> List[Any]:
        raise NotImplementedError

    def _capture_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns (residual_by_layer, mlp_by_layer) each a list of tensors with shape [B, S, D].
        """
        raise NotImplementedError

    def capture(
        self, prompts: List[str], batch_size: int = 8
    ) -> CaptureResult:
        """
        Capture last-token residual and MLP activations for all layers.
        """
        residual_chunks: List[np.ndarray] = []
        mlp_chunks: List[np.ndarray] = []
        last_idx_chunks: List[int] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            last_idx = attn.sum(dim=1) - 1
            with torch.no_grad():
                res_layers, mlp_layers = self._capture_batch(input_ids, attn)
            # Convert to numpy pooled last token per sample
            res_stack = torch.stack(res_layers, dim=1)  # [B, L, S, H]
            mlp_stack = torch.stack(mlp_layers, dim=1)  # [B, L, S, d_mlp]
            bsz = input_ids.shape[0]
            arange = torch.arange(bsz, device=self.device)
            pooled_res = res_stack[arange, :, last_idx, :].detach().cpu().numpy()
            pooled_mlp = mlp_stack[arange, :, last_idx, :].detach().cpu().numpy()
            residual_chunks.append(pooled_res)
            mlp_chunks.append(pooled_mlp)
            last_idx_chunks.extend(last_idx.tolist())
        if residual_chunks:
            residual = np.concatenate(residual_chunks, axis=0)
            mlp = np.concatenate(mlp_chunks, axis=0)
        else:
            residual = np.zeros((0, len(self.list_layers()), 0), dtype=np.float32)
            mlp = np.zeros_like(residual)
        return CaptureResult(
            residual=residual.astype(np.float32),
            mlp=mlp.astype(np.float32),
            metadata={"adapter": self.adapter_name, "model_name": self.model_name},
        )
