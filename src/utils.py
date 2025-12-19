from __future__ import annotations

from datetime import datetime
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(override: Optional[str] = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def maybe_load_npz(path: Path) -> Optional[dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


############################################################
# Hashing and manifests
############################################################


def stable_hash_json(obj: Any) -> str:
    """
    Deterministic SHA256 over a JSON-serializable object.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_samples(samples: list[Any]) -> str:
    """
    Hash samples based on ordered sample_id and prompt_text.
    """
    serial = [(getattr(s, "sample_id", ""), getattr(s, "prompt_text", "")) for s in samples]
    return stable_hash_json(serial)


def hash_splits(config: Any) -> str:
    """
    Hash the templates and synonyms for each concept and split.
    """
    concepts_payload = {}
    for name, spec in getattr(config, "concepts", {}).items():
        concepts_payload[name] = {
            "templates": spec.template_families,
            "discovery_synonyms": spec.discovery_synonyms,
            "eval_synonyms": spec.eval_synonyms,
        }
    return stable_hash_json(concepts_payload)


def build_manifest(
    *,
    seed: int,
    config_snapshot: dict[str, Any],
    dataset_signature: str,
    split_signature: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "config": config_snapshot,
        "dataset_signature": dataset_signature,
        "split_signature": split_signature,
    }
