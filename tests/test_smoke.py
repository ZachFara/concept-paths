import pytest

from src import config as cfg
from src.geometry_runner import run_geometry
from src.config import ControlSpec, DataSpec


@pytest.mark.slow
def test_geometry_smoke(tmp_path):
    conf = cfg.ExperimentConfig.defaults()
    try:
        res = run_geometry(
            cfg=conf,
            data_spec=DataSpec(concept="sentiment", split="discovery", template_family=conf.data.template_family, seed=0, n_levels=2, n_per_level=1),
            control_spec=ControlSpec(),
            artifacts_dir=tmp_path,
            adapter="gpt2",
            model="sshleifer/tiny-gpt2",
            batch_size=2,
            use_cache=False,
            n_boot=5,
        )
    except OSError:
        pytest.skip("Tiny model unavailable offline")
    assert "pc1" in res and res["pc1"].size > 0
