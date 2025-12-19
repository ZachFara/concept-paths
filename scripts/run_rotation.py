"""
Deprecated shim: use `python -m src.cli run_all --config configs/default.yaml`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".mplconfig"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    print(
        "[DEPRECATED] Use `python -m src.cli run_all --config configs/default.yaml`. "
        "Running run_all now."
    )
    subprocess.check_call(
        [sys.executable, "-m", "src.cli", "run_all", "--config", str(ROOT / "configs/default.yaml")]
    )


if __name__ == "__main__":
    main()
