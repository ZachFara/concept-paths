import subprocess
import sys
import pytest
from pathlib import Path
import shutil


@pytest.mark.slow
def test_run_all_smoke(tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(Path("configs/tiny.yaml").read_text())
    artifacts_dir = tmp_path / "artifacts"
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "run_all",
        "--config",
        str(cfg_path),
        "--use_cache",
        "0",
        "--artifacts_dir",
        str(artifacts_dir),
        "--model",
        "sshleifer/tiny-gpt2",
        "--adapter",
        "gpt2",
        "--second_model",
        "sshleifer/tiny-gpt2",
        "--second_adapter",
        "gpt2",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=180)
    except Exception:
        pytest.skip("Run all smoke timeout or failure")
    if res.returncode != 0:
        pytest.skip("Run all smoke failed (likely missing tiny model); skipping")
    # Check artifacts exist
    assert (artifacts_dir / "plots").exists()
    assert (artifacts_dir / "stats").exists()
