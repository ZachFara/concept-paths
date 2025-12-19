from __future__ import annotations

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .adapters import GPT2Adapter, ModelAdapter, OPTAdapter
from .config import ControlSpec, DataSpec, ExperimentConfig, load_experiment_config
from .data import generate_samples
from .utils import atomic_save_npz, ensure_dir, hash_splits, maybe_load_npz


@dataclass(frozen=True)
class ActivationCache:
    residual: np.ndarray  # [N, L, H]
    mlp: np.ndarray  # [N, L, D]
    metadata: Dict[str, object]


def _select_adapter(adapter_name: str, model_name: str, local_files_only: bool = True, device: Optional[str] = None) -> ModelAdapter:
    adapter_name = adapter_name.lower()
    if adapter_name in ("gpt2", "distilgpt2"):
        return GPT2Adapter(model_name, device=device, local_files_only=local_files_only)
    if adapter_name in ("opt", "opt-125m"):
        return OPTAdapter(model_name, device=device, local_files_only=local_files_only)
    # default fallback: try GPT2Adapter
    return GPT2Adapter(model_name, device=device, local_files_only=local_files_only)


def _cache_path(
    artifacts_dir: Path,
    *,
    adapter_name: str,
    model_name: str,
    concept: str,
    split: str,
    template_family: str,
    seed: int,
    dataset_sig: str,
) -> Path:
    safe_model = model_name.replace("/", "__")
    safe_adapter = adapter_name.replace("/", "__")
    return (
        artifacts_dir
        / "activations"
        / f"{safe_adapter}__{safe_model}__{concept}__{split}__{template_family}__seed{seed}__ds{dataset_sig}.npz"
    )


def capture_activations(
    *,
    config: ExperimentConfig,
    data_spec: Optional[DataSpec] = None,
    control_spec: Optional[ControlSpec] = None,
    adapter_name: str = "gpt2",
    model_name: str = "distilgpt2",
    batch_size: int = 4,
    artifacts_dir: Path = Path("artifacts"),
    use_cache: bool = True,
    local_files_only: bool = True,
    device: Optional[str] = None,
) -> ActivationCache:
    data_spec = data_spec or config.data
    control_spec = control_spec or config.control
    samples, dataset_sig = generate_samples(config, data_spec=data_spec, control=control_spec)
    prompts = [s.prompt_text for s in samples]
    adapter = _select_adapter(adapter_name, model_name, local_files_only=local_files_only, device=device)

    cache_path = _cache_path(
        artifacts_dir,
        adapter_name=adapter.adapter_name,
        model_name=model_name,
        concept=data_spec.concept,
        split=data_spec.split,
        template_family=data_spec.template_family,
        seed=data_spec.seed,
        dataset_sig=dataset_sig,
    )

    if use_cache:
        loaded = maybe_load_npz(cache_path)
        if loaded is not None:
            return ActivationCache(
                residual=loaded["residual"],
                mlp=loaded["mlp"],
                metadata=json.loads(loaded["metadata"].item()),
            )

    capture = adapter.capture(prompts, batch_size=batch_size)
    metadata = {
        "adapter": adapter.adapter_name,
        "model_name": model_name,
        "concept": data_spec.concept,
        "split": data_spec.split,
        "template_family": data_spec.template_family,
        "seed": data_spec.seed,
        "dataset_signature": dataset_sig,
        "split_signature": hash_splits(config),
    }
    atomic_save_npz(
        cache_path,
        residual=capture.residual.astype(np.float32),
        mlp=capture.mlp.astype(np.float32),
        metadata=np.array([json.dumps(metadata)]),
    )
    return ActivationCache(residual=capture.residual, mlp=capture.mlp, metadata=metadata)
