from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Any, Optional

from dataclasses import asdict, is_dataclass

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


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def stable_hash_json(obj: Any) -> str:
    payload = json.dumps(
        _to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_samples(samples: list[Any]) -> str:
    normalized = []
    for sample in samples:
        item = _to_jsonable(sample)
        if isinstance(item, dict):
            metadata = item.get("metadata", None)
            if isinstance(metadata, dict) and "dataset_signature" in metadata:
                metadata = dict(metadata)
                metadata.pop("dataset_signature", None)
                item["metadata"] = metadata
        normalized.append(item)
    normalized = sorted(
        normalized,
        key=lambda x: x.get("sample_id", "") if isinstance(x, dict) else str(x),
    )
    return stable_hash_json(normalized)


def hash_splits(splits: dict[str, list[Any]]) -> dict[str, str]:
    split_hashes = {name: hash_samples(samples) for name, samples in splits.items()}
    overall = stable_hash_json(split_hashes)
    return {"splits": split_hashes, "overall": overall}
