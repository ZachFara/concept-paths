import glob
import os
import pytest

from src import config as cfg
from src import capture


@pytest.mark.slow
def test_capture_smoke(tmp_path):
    conf = cfg.ExperimentConfig.defaults()
    data_spec = cfg.DataSpec(concept="sentiment", split="discovery", template_family="main", n_per_level=1)
    try:
        _cache = capture.capture_activations(
            config=conf,
            data_spec=data_spec,
            adapter_name="gpt2",
            model_name="sshleifer/tiny-gpt2",
            artifacts_dir=tmp_path,
            batch_size=2,
            use_cache=False,
            local_files_only=False,
        )
    except OSError:
        pytest.skip("Model not available locally and download blocked")

    files = glob.glob(os.path.join(tmp_path, "activations", "*.npz"))
    assert files, "No activation cache written"
    # Sanity check keys
    assert _cache.residual.ndim == 3
    assert _cache.mlp.ndim == 3
