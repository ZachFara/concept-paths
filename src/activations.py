from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import save_npz
from .utils import ensure_dir, get_device, maybe_load_npz


@dataclass(frozen=True)
class ActivationCache:
    keys: list[str]
    acts: np.ndarray  # [n_samples, n_layers, hidden]


def _get_blocks(model: Any) -> list[Any]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError(f"Unsupported model structure: {type(model)}")


def _prepare_tokenizer(tokenizer: Any) -> Any:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lm(
    model_name: str,
    *,
    device: Optional[torch.device] = None,
    local_files_only: bool = True,
    device_override: Optional[str] = None,
) -> LanguageModel:
    if device is None:
        device = get_device(device_override)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer = _prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    model.to(device)
    return LanguageModel(model, tokenizer=tokenizer)


def ensure_scanned(lm: LanguageModel, example: Any) -> None:
    """
    Populate fake outputs to enable .output access during tracing.
    """
    if getattr(lm, "_concept_paths_scanned", False):
        return
    try:
        with lm.scan(example):
            pass
        getattr(_get_blocks(lm)[0].output, "shape", None)
        setattr(lm, "_concept_paths_scanned", True)
    except Exception:
        # fallback to CPU if MPS gives trouble
        lm._model.to("cpu")  # type: ignore[attr-defined]
        with lm.scan(example):
            pass
        setattr(lm, "_concept_paths_scanned", True)


def _value(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


@torch.no_grad()
def capture_last_token_residuals(
    lm: LanguageModel,
    prompts: list[str],
    *,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    desc: str = "capture",
) -> np.ndarray:
    import math
    from tqdm import tqdm

    if device is None:
        device = get_device()
    blocks = _get_blocks(lm)
    tokenizer = lm.tokenizer
    outputs: list[np.ndarray] = []
    if prompts:
        ensure_scanned(lm, prompts[0])

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        last_idx = attention_mask.sum(dim=1) - 1
        bsz = enc["input_ids"].shape[0]
        batch_arange = torch.arange(bsz, device=device)

        saved = []
        with lm.trace(enc):
            for block in blocks:
                saved.append(block.output[0].save())
        pooled_layers: list[torch.Tensor] = []
        for s in saved:
            hs = _value(s)
            if hs.ndim == 3 and hs.shape[0] != bsz and hs.shape[1] == bsz:
                hs = hs.transpose(0, 1)
            pooled = hs[batch_arange, last_idx]
            pooled_layers.append(pooled)
        acts = torch.stack(pooled_layers, dim=1).detach().to("cpu", dtype=torch.float32).numpy()
        outputs.append(acts)

    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, len(blocks), 0))


def cache_path(artifacts_dir: Path, split: str, model: str, seed: int) -> Path:
    safe_model = model.replace("/", "__")
    return artifacts_dir / "activations" / f"{split}__{safe_model}__seed{seed}.npz"


def build_or_load_activation_cache(
    lm: LanguageModel,
    *,
    keys: list[str],
    prompts: list[str],
    artifacts_dir: Path,
    split: str,
    model_name: str,
    seed: int,
    batch_size: int,
    device: Optional[torch.device] = None,
    force_recompute: bool = False,
) -> ActivationCache:
    ensure_dir(artifacts_dir)
    path = cache_path(artifacts_dir, split, model_name, seed)
    if not force_recompute:
        loaded = maybe_load_npz(path)
        if loaded is not None:
            loaded_keys = loaded["keys"].astype(str).tolist()
            if loaded_keys == keys:
                return ActivationCache(keys=loaded_keys, acts=loaded["acts"])
    acts = capture_last_token_residuals(
        lm,
        prompts,
        batch_size=batch_size,
        device=device,
        desc=f"capture:{split}",
    )
    save_npz(path, keys=np.array(keys, dtype=str), acts=acts.astype(np.float32))
    return ActivationCache(keys=keys, acts=acts.astype(np.float32))
