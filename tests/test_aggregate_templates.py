from src.config import ProjectConfig
from src.data import generate_samples_all_families


def test_generate_samples_all_families_marks_aggregated() -> None:
    cfg = ProjectConfig()
    samples = generate_samples_all_families(
        "sentiment",
        "discovery",
        seed=0,
        n_per_level=1,
        data_spec=cfg.data,
        control_spec=cfg.controls,
        concept_mode="sentiment",
        aggregated_family_name="aggregated-templates",
    )
    assert samples, "expected aggregated samples"
    families = {s.metadata.get("original_template_family") for s in samples}
    assert "adjective_clause" in families
    assert all(s.metadata.get("template_family") == "aggregated-templates" for s in samples)
