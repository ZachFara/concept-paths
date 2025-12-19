import pytest

from src import config as cfg
from src import capture
from src import ablation
from src.data import generate_samples
from src.capture import _select_adapter


@pytest.mark.slow
def test_ablation_smoke(tmp_path):
    conf = cfg.ExperimentConfig.defaults()
    data_spec = cfg.DataSpec(concept="sentiment", split="discovery", template_family="main", n_per_level=1)
    control = cfg.ControlSpec()
    samples, _ = generate_samples(conf, data_spec=data_spec, control=control)
    try:
        cache = capture.capture_activations(
            config=conf,
            data_spec=data_spec,
            control_spec=control,
            adapter_name="gpt2",
            model_name="sshleifer/tiny-gpt2",
            batch_size=2,
            artifacts_dir=tmp_path,
            use_cache=False,
            local_files_only=False,
        )
    except OSError:
        pytest.skip("Model not available locally")
    adapter = _select_adapter("gpt2", "sshleifer/tiny-gpt2", local_files_only=False)
    out_dir = tmp_path / "ablation"
    ablation.run_ablation_pipeline(
        adapter=adapter,
        samples=samples,
        mlp=cache.mlp,
        residual=cache.residual,
        layer=0,
        m_list=[1],
        methods=["variance"],
        seed=0,
        out_dir=out_dir,
    )
    assert (out_dir / "dose_variance.png").exists()
