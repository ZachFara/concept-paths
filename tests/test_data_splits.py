from collections import Counter

from src.config import ControlSpec, DataSpec
from src.data import (
    apply_random_label_control,
    apply_unrelated_concept_control,
    generate_samples,
)


def test_templates_and_synonyms_disjoint() -> None:
    data_spec = DataSpec()
    for concept in data_spec.concepts.values():
        for level in concept.levels:
            disc = concept.synonyms["discovery"][level]
            eval_ = concept.synonyms["eval"][level]
            assert set(disc).isdisjoint(set(eval_))
        for family, splits in concept.templates.items():
            disc = splits["discovery"]
            eval_ = splits["eval"]
            assert set(disc).isdisjoint(set(eval_))

    controls = ControlSpec()
    assert set(controls.neutral_templates["discovery"]).isdisjoint(
        set(controls.neutral_templates["eval"])
    )


def _counts_by_level(samples):
    return Counter(s.level for s in samples)


def _counts_by_index(samples, levels):
    index = {level: i for i, level in enumerate(levels)}
    return Counter(index[s.level] for s in samples)


def test_controls_preserve_counts_and_prompts() -> None:
    base = generate_samples(
        "sentiment",
        "discovery",
        "adjective_clause",
        seed=0,
        n_per_level=2,
    )
    base_prompts = [s.prompt_text for s in base]

    random = apply_random_label_control(base, seed=0)
    assert base_prompts == [s.prompt_text for s in random]
    assert _counts_by_level(base) == _counts_by_level(random)

    unrelated = apply_unrelated_concept_control(base, "concreteness")
    assert base_prompts == [s.prompt_text for s in unrelated]

    base_levels = DataSpec().concepts["sentiment"].levels
    target_levels = DataSpec().concepts["concreteness"].levels
    assert _counts_by_index(base, base_levels) == _counts_by_index(unrelated, target_levels)
