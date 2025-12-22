from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import (
    atomic_save_npz,
    ensure_dir,
    get_device,
    load_json,
    maybe_load_npz,
    save_json,
    stable_hash_json,
)

try:
    from nnsight import LanguageModel
except Exception as exc:  # pragma: no cover - optional dependency
    LanguageModel = None  # type: ignore[assignment]
    _NNSIGHT_IMPORT_ERROR = exc
else:
    _NNSIGHT_IMPORT_ERROR = None

CaptureSite = Literal["residual", "mlp"]


@dataclass(frozen=True)
class ActivationCache:
    keys: list[str]
    residual: np.ndarray  # [layers, samples, d_model]
    mlp: np.ndarray  # [layers, samples, d_mlp]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    tokenizer: Any
    model: Any
    lm: Any
    device: torch.device


@dataclass(frozen=True)
class CaptureDataset:
    keys: list[str]
    prompts: list[str]
    concept: str
    split: str
    template_family: str
    seed: int
    control_flags: dict[str, Any]
    dataset_signature: str


class GPT2Adapter:
    family = "gpt2"

    def blocks(self, model_or_lm: Any) -> list[Any]:
        if hasattr(model_or_lm, "transformer") and hasattr(model_or_lm.transformer, "h"):
            return list(model_or_lm.transformer.h)
        model = _unwrap_model(model_or_lm)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        raise ValueError("GPT-2 adapter could not locate transformer blocks")

    def block_out(self, block: Any) -> Any:
        return block.output[0]

    def ffn_out(self, block: Any) -> Any:
        return block.mlp.c_proj.output


class OPTAdapter:
    family = "opt"

    def blocks(self, model_or_lm: Any) -> list[Any]:
        if hasattr(model_or_lm, "model") and hasattr(model_or_lm.model, "decoder") and hasattr(
            model_or_lm.model.decoder, "layers"
        ):
            return list(model_or_lm.model.decoder.layers)
        model = _unwrap_model(model_or_lm)
        if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(
            model.model.decoder, "layers"
        ):
            return list(model.model.decoder.layers)
        raise ValueError("OPT adapter could not locate decoder layers")

    def block_out(self, block: Any) -> Any:
        return block.output[0]

    def ffn_out(self, block: Any) -> Any:
        return block.fc2.output


def _unwrap_model(model_or_lm: Any) -> Any:
    if hasattr(model_or_lm, "_model"):
        return model_or_lm._model  # type: ignore[attr-defined]
    return model_or_lm


def resolve_adapter(model_or_lm: Any, model_name: str | None = None) -> GPT2Adapter | OPTAdapter:
    model = _unwrap_model(model_or_lm)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    model_name = model_name or getattr(getattr(model, "config", None), "name_or_path", None)
    if model_type == "gpt2" or (model_name and "gpt2" in model_name):
        return GPT2Adapter()
    if model_type == "opt" or (model_name and "opt" in model_name):
        return OPTAdapter()
    raise ValueError(f"Unsupported model family: model_type={model_type}, name={model_name}")


def _prepare_tokenizer(tokenizer: Any) -> Any:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_bundle(
    model_name: str,
    *,
    device: torch.device | None = None,
    local_files_only: bool = True,
) -> ModelBundle:
    if LanguageModel is None:
        raise RuntimeError(
            "nnsight is not available; install it before running capture"
        ) from _NNSIGHT_IMPORT_ERROR
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer = _prepare_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    model.to(device)

    lm = LanguageModel(model, tokenizer=tokenizer)
    return ModelBundle(
        model_name=model_name,
        tokenizer=tokenizer,
        model=model,
        lm=lm,
        device=device,
    )


def _value(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def _get_blocks(model_or_lm: Any) -> list[Any]:
    adapter = resolve_adapter(model_or_lm)
    return adapter.blocks(model_or_lm)


def ensure_scanned(lm: Any, example: Any) -> None:
    if getattr(lm, "_concept_paths_scanned", False):
        return

    def _validate_scan() -> None:
        blocks = _get_blocks(lm)
        _ = blocks[0].output[0].shape  # noqa: F841

    try:
        with lm.scan(example):
            pass
        _validate_scan()
        setattr(lm, "_concept_paths_scanned", True)
        return
    except Exception as first_exc:
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


def resolve_device(lm: Any, device: torch.device) -> torch.device:
    override = getattr(lm, "_concept_paths_device_override", None)
    return override if override is not None else device


def _ensure_bsh(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    if t.ndim == 3 and t.shape[0] != batch_size and t.shape[1] == batch_size:
        return t.transpose(0, 1)
    return t


def _pool_last_token(
    h: torch.Tensor,
    *,
    last_idx: torch.Tensor,
    batch_arange: torch.Tensor,
) -> torch.Tensor:
    return h[batch_arange, last_idx]


@torch.no_grad()
def capture_activations(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    batch_size: int = 16,
    pooling: str = "last",
    capture_sites: Sequence[CaptureSite] = ("residual", "mlp"),
    desc: str = "capture",
) -> dict[CaptureSite, np.ndarray]:
    """
    Capture pooled residual stream (block output) and MLP output per layer.

    Contract:
      - residual: [layers, samples, d_model]
      - mlp: [layers, samples, d_mlp] (GPT-2: c_proj output, OPT: fc2 output)
    """
    if pooling != "last":
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    adapter = resolve_adapter(bundle.lm, model_name=bundle.model_name)
    tokenizer = bundle.tokenizer

    if prompts:
        ensure_scanned(bundle.lm, prompts[0])

    blocks = adapter.blocks(bundle.lm)
    n_layers = len(blocks)

    outputs: dict[CaptureSite, list[np.ndarray]] = {"residual": [], "mlp": []}

    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[start : start + batch_size]
        device = resolve_device(bundle.lm, bundle.device)
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        attention_mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))
        last_idx = attention_mask.sum(dim=1) - 1
        bsz = enc["input_ids"].shape[0]
        batch_arange = torch.arange(bsz, device=device)

        saved: dict[CaptureSite, list[Any]] = {"residual": [], "mlp": []}
        with bundle.lm.trace(enc):
            for block in blocks:
                if "mlp" in capture_sites:
                    saved["mlp"].append(adapter.ffn_out(block).save())
                if "residual" in capture_sites:
                    saved["residual"].append(adapter.block_out(block).save())

        for site in capture_sites:
            pooled_layers: list[torch.Tensor] = []
            seq_len = enc["input_ids"].shape[1]
            for h in saved[site]:
                h_t = _ensure_bsh(_value(h), bsz)
                if h_t.ndim == 2:
                    if h_t.shape[0] == bsz * seq_len:
                        h_t = h_t.reshape(bsz, seq_len, -1)
                        pooled = _pool_last_token(h_t, last_idx=last_idx, batch_arange=batch_arange)
                    elif h_t.shape[0] == seq_len and bsz == 1:
                        h_t = h_t.unsqueeze(0)
                        pooled = _pool_last_token(h_t, last_idx=last_idx, batch_arange=batch_arange)
                    elif h_t.shape[0] != bsz and h_t.shape[1] == bsz:
                        h_t = h_t.transpose(0, 1)
                        pooled = h_t
                    elif h_t.shape[0] == bsz:
                        pooled = h_t
                    else:
                        raise ValueError(
                            f"Unexpected 2D activation shape {h_t.shape} for batch {bsz}"
                        )
                elif h_t.ndim == 3:
                    pooled = _pool_last_token(h_t, last_idx=last_idx, batch_arange=batch_arange)
                else:
                    raise ValueError(f"Expected activation tensor with 2 or 3 dims, got {h_t.shape}")
                pooled_layers.append(pooled)
            acts = (
                torch.stack(pooled_layers, dim=0)
                .detach()
                .to("cpu", dtype=torch.float32)
                .numpy()
            )
            outputs[site].append(acts)

    result: dict[CaptureSite, np.ndarray] = {}
    for site in capture_sites:
        if outputs[site]:
            result[site] = np.concatenate(outputs[site], axis=1)
        else:
            result[site] = np.zeros((n_layers, 0, 0), dtype=np.float32)
    return result


def _dataset_signature_from_keys(keys: Sequence[str], prompts: Sequence[str]) -> str:
    payload = [{"key": k, "prompt": p} for k, p in zip(keys, prompts, strict=True)]
    return stable_hash_json(payload)


def dataset_from_samples(samples: Sequence[Any]) -> CaptureDataset:
    if not samples:
        raise ValueError("samples required")
    first = samples[0]
    metadata = getattr(first, "metadata", {}) or {}
    concept = getattr(first, "concept_name", "unknown")
    split = metadata.get("split", "unknown")
    template_family = metadata.get("template_family", "unknown")
    seed = int(metadata.get("seed", 0))
    control_flags = {
        k: v
        for k, v in metadata.items()
        if k == "control"
        or k.startswith("original_")
        or k in {"concept_mode", "topics_signature"}
    }
    dataset_signature = metadata.get("dataset_signature")
    keys = [getattr(s, "sample_id", "") for s in samples]
    prompts = [getattr(s, "prompt_text", "") for s in samples]
    if not dataset_signature:
        dataset_signature = _dataset_signature_from_keys(keys, prompts)
    return CaptureDataset(
        keys=keys,
        prompts=prompts,
        concept=concept,
        split=split,
        template_family=template_family,
        seed=seed,
        control_flags=control_flags,
        dataset_signature=dataset_signature,
    )


def dataset_from_prompts(
    *,
    keys: Sequence[str],
    prompts: Sequence[str],
    concept: str,
    split: str,
    template_family: str,
    seed: int,
    control_flags: dict[str, Any] | None = None,
    dataset_signature: str | None = None,
) -> CaptureDataset:
    if dataset_signature is None:
        dataset_signature = _dataset_signature_from_keys(keys, prompts)
    return CaptureDataset(
        keys=list(keys),
        prompts=list(prompts),
        concept=concept,
        split=split,
        template_family=template_family,
        seed=seed,
        control_flags=control_flags or {},
        dataset_signature=dataset_signature,
    )


def _cache_key(
    dataset: CaptureDataset,
    *,
    model_name: str,
    pooling: str,
    capture_sites: Sequence[CaptureSite],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "concept": dataset.concept,
        "split": dataset.split,
        "template_family": dataset.template_family,
        "control_flags": dataset.control_flags,
        "seed": dataset.seed,
        "dataset_signature": dataset.dataset_signature,
        "capture_site": list(capture_sites),
        "pooling": pooling,
    }


def _cache_paths(
    artifacts_dir: Path,
    *,
    dataset: CaptureDataset,
    model_name: str,
    pooling: str,
    capture_sites: Sequence[CaptureSite],
) -> tuple[Path, Path, dict[str, Any], str]:
    safe_model = model_name.replace("/", "__")
    cache_key = _cache_key(
        dataset,
        model_name=model_name,
        pooling=pooling,
        capture_sites=capture_sites,
    )
    cache_hash = stable_hash_json(cache_key)
    stem = (
        f"{dataset.split}__{dataset.concept}__{dataset.template_family}__"
        f"{safe_model}__{pooling}__seed{dataset.seed}__{cache_hash[:10]}"
    )
    base = artifacts_dir / "activations" / stem
    return base.with_suffix(".npz"), base.with_suffix(".json"), cache_key, cache_hash


def build_or_load_activation_cache(
    bundle: ModelBundle,
    *,
    dataset: CaptureDataset,
    artifacts_dir: Path,
    batch_size: int,
    pooling: str = "last",
    capture_sites: Sequence[CaptureSite] = ("residual", "mlp"),
    use_cache: bool = True,
) -> ActivationCache:
    ensure_dir(artifacts_dir)
    npz_path, meta_path, cache_key, cache_hash = _cache_paths(
        artifacts_dir,
        dataset=dataset,
        model_name=bundle.model_name,
        pooling=pooling,
        capture_sites=capture_sites,
    )

    if use_cache:
        loaded = maybe_load_npz(npz_path)
        if loaded is not None and meta_path.exists():
            meta_obj = load_json(meta_path)
            loaded_keys = loaded.get("keys")
            if loaded_keys is not None:
                loaded_keys = loaded_keys.astype(str).tolist()
                if loaded_keys == dataset.keys and meta_obj.get("cache_hash") == cache_hash:
                    residual = loaded.get("residual")
                    mlp = loaded.get("mlp")
                    if residual is None or mlp is None:
                        raise ValueError("Cached activations missing residual or mlp arrays")
                    return ActivationCache(
                        keys=loaded_keys,
                        residual=residual.astype(np.float32),
                        mlp=mlp.astype(np.float32),
                        metadata=meta_obj,
                    )

    acts = capture_activations(
        bundle,
        dataset.prompts,
        batch_size=batch_size,
        pooling=pooling,
        capture_sites=capture_sites,
        desc=f"capture:{dataset.split}",
    )
    residual = acts.get("residual")
    mlp = acts.get("mlp")
    if residual is None or mlp is None:
        raise ValueError("Capture missing required residual or mlp outputs")

    metadata = {
        "cache_key": cache_key,
        "cache_hash": cache_hash,
        "model_name": bundle.model_name,
        "pooling": pooling,
        "capture_sites": list(capture_sites),
        "residual_site": "block_out",
        "mlp_site": "ffn_out",
        "concept": dataset.concept,
        "split": dataset.split,
        "template_family": dataset.template_family,
        "seed": dataset.seed,
        "control_flags": dataset.control_flags,
        "dataset_signature": dataset.dataset_signature,
        "n_samples": len(dataset.keys),
        "n_layers": int(residual.shape[0]),
        "residual_dim": int(residual.shape[2]) if residual.size else 0,
        "mlp_dim": int(mlp.shape[2]) if mlp.size else 0,
    }
    save_json(meta_path, metadata)
    atomic_save_npz(
        npz_path,
        keys=np.array(dataset.keys, dtype=str),
        residual=residual.astype(np.float32),
        mlp=mlp.astype(np.float32),
    )
    return ActivationCache(
        keys=dataset.keys,
        residual=residual.astype(np.float32),
        mlp=mlp.astype(np.float32),
        metadata=metadata,
    )
