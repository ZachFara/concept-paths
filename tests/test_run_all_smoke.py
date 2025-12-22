from pathlib import Path

import yaml

from src.cli import cmd_run_all


def test_run_all_smoke(tmp_path: Path) -> None:
    cfg = {
        "model_name": "distilgpt2",
        "local_files_only": True,
        "batch_size": 4,
        "seeds": [0],
        "artifacts_dir": str(tmp_path / "artifacts"),
        "model_names": ["distilgpt2"],
        "concepts": ["sentiment"],
        "use_cache": True,
        "n_per_level": 1,
        "geometry": {"n_bootstrap": 2},
        "specificity": {"n_shuffles": 2},
        "ablation": {"layer": 0, "method": "variance", "m_list": [5]},
        "behavior": {"ablate_layer": 0, "m": 5, "method": "probe_weight"},
    }
    cfg_path = tmp_path / "paper.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    args = type("Args", (), {"config": str(cfg_path), "backend": "nnsight", "use_cache": 1})()
    cmd_run_all(args)

    run_dirs = [p for p in (tmp_path / "artifacts").iterdir() if p.is_dir()]
    assert run_dirs
    run_dir = next(p for p in run_dirs if (p / "plots").exists())
    assert (run_dir / "plots").exists()
    assert (run_dir / "stats").exists()
    assert (run_dir / "paper_figures").exists()
    assert (run_dir / "index.json").exists()
