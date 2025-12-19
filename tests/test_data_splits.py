from collections import Counter

from src import config as cfg
from src import data as dat


def _base_config() -> cfg.ExperimentConfig:
    return cfg.ExperimentConfig.defaults()


def test_templates_disjoint_main_and_control():
    conf = _base_config()
    for name, spec in conf.concepts.items():
        for family in ("main", "control"):
            disc = set(spec.template_families[family]["discovery"])
            eval_ = set(spec.template_families[family]["eval"])
            assert disc.isdisjoint(eval_), f"Templates overlap for {name}, family {family}"


def test_synonyms_disjoint_by_split():
    conf = _base_config()
    for name, spec in conf.concepts.items():
        for lvl in spec.levels:
            disc = set(spec.discovery_synonyms.get(lvl, []))
            eval_ = set(spec.eval_synonyms.get(lvl, []))
            assert disc.isdisjoint(eval_), f"Synonyms overlap for {name}, level {lvl}"


def test_random_label_shuffle_preserves_counts_and_prompts():
    conf = _base_config()
    base_samples, _ = dat.generate_samples(conf)
    control = cfg.ControlSpec(random_labels=True)
    shuffled, _ = dat.generate_samples(conf, control=control)

    base_prompts = [s.prompt_text for s in base_samples]
    shuf_prompts = [s.prompt_text for s in shuffled]
    assert base_prompts == shuf_prompts  # prompts unchanged

    base_counts = Counter(s.level for s in base_samples)
    shuf_counts = Counter(s.level for s in shuffled)
    assert base_counts == shuf_counts


def test_unrelated_labels_preserve_counts_and_prompts():
    conf = _base_config()
    base_samples, _ = dat.generate_samples(conf)
    control = cfg.ControlSpec(unrelated_labels=True)
    unrelated, _ = dat.generate_samples(conf, control=control)

    base_prompts = [s.prompt_text for s in base_samples]
    unrelated_prompts = [s.prompt_text for s in unrelated]
    assert base_prompts == unrelated_prompts

    base_counts = Counter(s.level for s in base_samples)
    unrel_counts = Counter(s.level for s in unrelated)
    assert base_counts == unrel_counts
