import numpy as np
from src.ablation import run_ablation
from src.config import ProjectConfig
from src.data import generate_samples


def test_ablation_smoke() -> None:
    cfg = ProjectConfig()
    concept = "sentiment"
    template_family = list(cfg.data.concepts[concept].templates.keys())[0]
    samples = generate_samples(
        concept,
        "eval",
        template_family,
        seed=0,
        n_per_level=1,
        data_spec=cfg.data,
        control_spec=cfg.controls,
    )
    result = run_ablation(
        samples,
        model_name="distilgpt2",
        selection_method="variance",
        layer=0,
        m_list=[5],
        alpha=0.1,
        random_control=False,
        batch_size=4,
        seed=0,
    )
    summary = result.summary
    assert "auc" in summary
    assert np.isfinite(summary["auc"])
    assert len(summary["effect_means"]) == 1
    assert np.isfinite(summary["effect_means"][0])
