from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import yaml

from .utils import ensure_dir, save_json


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def hash_prompts(prompts: Iterable[str]) -> list[str]:
    return [hash_prompt(p) for p in prompts]


def write_manifest(path: Path, data: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def save_config_snapshot(run_dir: Path, cfg_obj: Any) -> None:
    ensure_dir(run_dir)
    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    data = _to_jsonable(asdict(cfg_obj))
    json_path = run_dir / "config.json"
    yaml_path = run_dir / "config.yaml"
    with json_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    with yaml_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def package_versions() -> Dict[str, str]:
    pkgs = {}
    for name in ["torch", "transformers", "nnsight", "numpy", "scipy", "sklearn", "matplotlib"]:
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[name] = "not_installed"
    return pkgs


def write_run_metadata(run_dir: Path, cfg_obj: Any, *, seeds: list[int]) -> None:
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "seeds": seeds,
    }
    write_manifest(run_dir / "run_metadata.json", meta)


def ensure_run_dir(artifacts_root: Path, run_id: Optional[str]) -> Path:
    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_root / run_id
    ensure_dir(run_dir)
    return run_dir


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
