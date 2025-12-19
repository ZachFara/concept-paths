import pytest

from src import config as cfgmod
from src.experiments.pipeline import run_geometry


@pytest.mark.slow
def test_geometry_smoke():
    cfg = cfgmod.RunConfig(
        seeds=[0],
        batch_size=4,
        geometry=cfgmod.GeometryConfig(delta_pair_strategy="random", random_baseline_directions=1, random_baseline_subspaces=1),
    )
    agg = run_geometry(cfg, permute_labels=False, concept_mode="sentiment", run_dir=cfg.artifacts_dir / "smoke")
    # Basic sanity: curves present and finite
    for split in ["discovery", "eval"]:
        assert agg[split]["k90_mean"].shape[0] > 0
