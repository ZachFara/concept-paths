from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import atomic_save_npz, ensure_dir, get_device, maybe_load_npz


@dataclass(frozen=True)
class ActivationCache:
    keys: list[str]  # aligns with rows in acts
    acts: np.ndarray  # [n_samples, n_layers, hidden]


def _value(x: Any) -> Any:
    # nnsight versions differ: `.save()` may return a wrapper with `.value` or the Tensor itself.
    return x.value if hasattr(x, "value") else x

def ensure_scanned(lm: LanguageModel, example: Any) -> None:
    """
    nnsight requires a .scan() pass to populate fake outputs so that `.output[...]`
    can be referenced inside later traces.
    """
    if getattr(lm, "_concept_paths_scanned", False):
        return

    def _validate_scan() -> None:
        blocks = _get_blocks(lm)
        # Accessing `.output` outside trace should work after scan (fake outputs populated).
        _ = blocks[0].output[0].shape  # noqa: F841

    try:
        with lm.scan(example):
            pass
        _validate_scan()
        setattr(lm, "_concept_paths_scanned", True)
        return
    except Exception as first_exc:
        # Empirically, FakeTensorMode/scan can be brittle on MPS. Fall back to CPU.
        try:
            device_type = next(lm._model.parameters()).device.type  # type: ignore[attr-defined]
        except StopIteration:
            device_type = "unknown"

        if device_type == "mps":
            lm._model.to("cpu")  # type: ignore[attr-defined]
            setattr(lm, "_concept_paths_device_override", torch.device("cpu"))
            with lm.scan(example):
                pass
            _validate_scan()
            setattr(lm, "_concept_paths_scanned", True)
            return

        raise RuntimeError("nnsight scan failed; try running on CPU.") from first_exc


def resolve_device(lm: LanguageModel, device: torch.device) -> torch.device:
    override = getattr(lm, "_concept_paths_device_override", None)
    return override if override is not None else device


def _get_blocks(model: Any) -> list[Any]:
    # nnsight LanguageModel exposes transformer submodules directly on the wrapper.
    if hasattr(model, "_model") and hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # LLaMA style (AutoModelForCausalLM wraps base at .model)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError(f"Unsupported model structure: {type(model)}")


def _prepare_tokenizer(tokenizer: Any) -> Any:
    # GPT-2 has no pad token by default; set it to eos for batching/padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_lm(
    model_name: str,
    *,
    device: torch.device | None = None,
    local_files_only: bool = True,
) -> LanguageModel:
    """
    Load a HF causal LM and wrap it in an nnsight LanguageModel.

    We load the model/tokenizer via transformers with local_files_only to avoid network calls,
    then wrap the loaded objects (so nnsight doesn't hit the hub).
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer = _prepare_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    model.to(device)

    return LanguageModel(model, tokenizer=tokenizer)


@torch.no_grad()
def capture_last_token_residuals(
    lm: LanguageModel,
    prompts: list[str],
    *,
    batch_size: int = 16,
    device: torch.device | None = None,
    desc: str = "capture",
) -> np.ndarray:
    """
    Returns pooled (last non-pad token) residual stream activations at each transformer block.

    Output: [n_prompts, n_layers, hidden] float32 on CPU as numpy.
    """
    if device is None:
        device = get_device()

    blocks = _get_blocks(lm)
    n_layers = len(blocks)

    tokenizer = lm.tokenizer
    out_chunks: list[np.ndarray] = []

    if prompts:
        # Scan using a representative input to enable `.output` access.
        ensure_scanned(lm, prompts[0])

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is None:
            # Should not happen with HF tokenizers, but keep a safe fallback.
            attention_mask = torch.ones_like(enc["input_ids"])

        # Last *non-pad* token per sample
        last_idx = attention_mask.sum(dim=1) - 1  # [B]
        batch_size_eff = enc["input_ids"].shape[0]

        saved = []
        with lm.trace(enc):
            # Avoid list comprehensions inside nnsight trace blocks; they execute in
            # their own frame and can confuse the tracer's control flow.
            for block in blocks:
                saved.append(block.output[0].save())

        pooled_layers: list[torch.Tensor] = []
        batch_arange = torch.arange(batch_size_eff, device=device)
        for s in saved:
            hs = _value(s)  # [B, S, H]
            pooled = hs[batch_arange, last_idx]  # [B, H]
            pooled_layers.append(pooled)

        # [B, L, H]
        acts = torch.stack(pooled_layers, dim=1).detach().to("cpu", dtype=torch.float32).numpy()
        out_chunks.append(acts)

    return np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0, n_layers, 0))


def get_activation_cache_path(
    artifacts_dir: Path,
    *,
    split: str,
    model_name: str,
    seed: int,
) -> Path:
    safe_model = model_name.replace("/", "__")
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
    device: torch.device | None = None,
    force_recompute: bool = False,
) -> ActivationCache:
    """
    Cache activations on disk to avoid recomputation.
    """
    ensure_dir(artifacts_dir)
    path = get_activation_cache_path(artifacts_dir, split=split, model_name=model_name, seed=seed)

    if not force_recompute:
        loaded = maybe_load_npz(path)
        if loaded is not None:
            loaded_keys = loaded["keys"].astype(str).tolist()
            acts = loaded["acts"]
            if loaded_keys == keys:
                return ActivationCache(keys=loaded_keys, acts=acts)

    acts = capture_last_token_residuals(
        lm,
        prompts,
        batch_size=batch_size,
        device=device,
        desc=f"capture:{split}",
    )
    atomic_save_npz(path, keys=np.array(keys, dtype=str), acts=acts.astype(np.float32))
    return ActivationCache(keys=keys, acts=acts.astype(np.float32))
