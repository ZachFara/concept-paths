from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Literal, Sequence

import numpy as np

from .config import ControlSpec, DataSpec, ProjectConfig
from .utils import hash_samples, stable_hash_json

Split = Literal["discovery", "eval"]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    concept_name: str
    level: str
    template_id: int
    synonym: str
    prompt_text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class DeltaPair:
    split: Split
    template_id: int
    neg_level_id: int
    pos_level_id: int
    neg_word: str
    pos_word: str
    neg_key: str
    pos_key: str


def _assert_disjoint(a: Iterable[str], b: Iterable[str], label: str) -> None:
    set_a = set(a)
    set_b = set(b)
    overlap = set_a.intersection(set_b)
    if overlap:
        raise ValueError(f"Leakage detected: {label} overlap: {sorted(overlap)[:5]}")


def _get_templates(
    concept_name: str,
    template_family: str,
    split: Split,
    data_spec: DataSpec,
    control_spec: ControlSpec,
) -> list[str]:
    if template_family == control_spec.neutral_template_family:
        return list(control_spec.neutral_templates[split])
    concept = data_spec.concepts[concept_name]
    if template_family not in concept.templates:
        raise KeyError(f"Unknown template_family '{template_family}' for {concept_name}")
    return list(concept.templates[template_family][split])


def _template_has_topic(templates: Sequence[str]) -> bool:
    return any("{topic}" in template for template in templates)


def _validate_disjointness(
    concept_name: str,
    data_spec: DataSpec,
    control_spec: ControlSpec,
    template_family: str,
) -> None:
    concept = data_spec.concepts[concept_name]
    for level in concept.levels:
        _assert_disjoint(
            concept.synonyms["discovery"][level],
            concept.synonyms["eval"][level],
            f"synonyms level {concept_name}.{level}",
        )
    if template_family == control_spec.neutral_template_family:
        _assert_disjoint(
            control_spec.neutral_templates["discovery"],
            control_spec.neutral_templates["eval"],
            f"templates family {control_spec.neutral_template_family}",
        )
    else:
        families = concept.templates[template_family]
        _assert_disjoint(
            families["discovery"],
            families["eval"],
            f"templates family {concept_name}.{template_family}",
        )
        if _template_has_topic(families["discovery"]) or _template_has_topic(families["eval"]):
            _assert_disjoint(
                control_spec.topic_swap_topics["discovery"],
                control_spec.topic_swap_topics["eval"],
                "topics list",
            )


def _select_synonyms(
    synonyms: Sequence[str], n_per_level: int, rng: np.random.Generator
) -> list[str]:
    if n_per_level > len(synonyms):
        raise ValueError(
            f"n_per_level={n_per_level} exceeds available synonyms={len(synonyms)}"
        )
    if n_per_level == len(synonyms):
        return list(synonyms)
    idx = rng.choice(len(synonyms), size=n_per_level, replace=False)
    return [synonyms[int(i)] for i in idx]


def _attach_dataset_signature(samples: list[Sample]) -> list[Sample]:
    signature = hash_samples(samples)
    updated: list[Sample] = []
    for s in samples:
        metadata = dict(s.metadata)
        metadata["dataset_signature"] = signature
        updated.append(
            Sample(
                sample_id=s.sample_id,
                concept_name=s.concept_name,
                level=s.level,
                template_id=s.template_id,
                synonym=s.synonym,
                prompt_text=s.prompt_text,
                metadata=metadata,
            )
        )
    return updated


def generate_samples(
    concept_name: str,
    split: Split,
    template_family: str,
    seed: int,
    n_per_level: int,
    *,
    data_spec: DataSpec | None = None,
    control_spec: ControlSpec | None = None,
    concept_mode: str = "sentiment",
) -> list[Sample]:
    data_spec = data_spec or ProjectConfig().data
    control_spec = control_spec or ProjectConfig().controls
    if split not in data_spec.split_names:
        raise ValueError(f"Unknown split: {split}")
    if concept_name not in data_spec.concepts:
        raise KeyError(f"Unknown concept: {concept_name}")

    _validate_disjointness(concept_name, data_spec, control_spec, template_family)

    concept = data_spec.concepts[concept_name]
    templates = _get_templates(concept_name, template_family, split, data_spec, control_spec)
    synonyms = concept.synonyms[split]
    topics = None
    if _template_has_topic(templates):
        topics = list(control_spec.topic_swap_topics[split])
    topics_signature = stable_hash_json(topics) if topics else None

    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for template_id, template in enumerate(templates):
        for level_id, level in enumerate(concept.levels):
            words = _select_synonyms(synonyms[level], n_per_level, rng)
            for word in words:
                if topics:
                    for topic_id, topic in enumerate(topics):
                        prompt_text = template.format(w=word, topic=topic)
                        sample_id = (
                            f"{concept_name}|{split}|{template_family}|{template_id}|"
                            f"{level}|{word}|{topic}"
                        )
                        metadata = {
                            "split": split,
                            "template_family": template_family,
                            "template": template,
                            "seed": seed,
                            "n_per_level": n_per_level,
                            "level_id": level_id,
                            "topic": topic,
                            "topic_id": topic_id,
                            "concept_mode": concept_mode,
                        }
                        if topics_signature is not None:
                            metadata["topics_signature"] = topics_signature
                        samples.append(
                            Sample(
                                sample_id=sample_id,
                                concept_name=concept_name,
                                level=level,
                                template_id=template_id,
                                synonym=word,
                                prompt_text=prompt_text,
                                metadata=metadata,
                            )
                        )
                else:
                    prompt_text = template.format(w=word)
                    sample_id = (
                        f"{concept_name}|{split}|{template_family}|{template_id}|{level}|{word}"
                    )
                    metadata = {
                        "split": split,
                        "template_family": template_family,
                        "template": template,
                        "seed": seed,
                        "n_per_level": n_per_level,
                        "level_id": level_id,
                        "concept_mode": concept_mode,
                    }
                    samples.append(
                        Sample(
                            sample_id=sample_id,
                            concept_name=concept_name,
                            level=level,
                            template_id=template_id,
                            synonym=word,
                            prompt_text=prompt_text,
                            metadata=metadata,
                        )
                    )
    return _attach_dataset_signature(samples)


def generate_samples_all_families(
    concept_name: str,
    split: Split,
    seed: int,
    n_per_level: int,
    *,
    data_spec: DataSpec | None = None,
    control_spec: ControlSpec | None = None,
    concept_mode: str = "sentiment",
    aggregated_family_name: str = "aggregated-templates",
) -> list[Sample]:
    data_spec = data_spec or ProjectConfig().data
    control_spec = control_spec or ProjectConfig().controls
    concept = data_spec.concepts[concept_name]
    all_samples: list[Sample] = []
    for family in sorted(concept.templates.keys()):
        fam_samples = generate_samples(
            concept_name,
            split,
            family,
            seed=seed,
            n_per_level=n_per_level,
            data_spec=data_spec,
            control_spec=control_spec,
            concept_mode=concept_mode,
        )
        all_samples.extend(fam_samples)

    updated: list[Sample] = []
    for s in all_samples:
        metadata = dict(s.metadata)
        metadata["original_template_family"] = metadata.get("template_family")
        metadata["template_family"] = aggregated_family_name
        updated.append(
            Sample(
                sample_id=s.sample_id,
                concept_name=s.concept_name,
                level=s.level,
                template_id=s.template_id,
                synonym=s.synonym,
                prompt_text=s.prompt_text,
                metadata=metadata,
            )
        )
    return _attach_dataset_signature(updated)


def apply_random_label_control(
    samples: Sequence[Sample],
    seed: int,
) -> list[Sample]:
    if not samples:
        return []
    rng = np.random.default_rng(seed)
    labels = [s.level for s in samples]
    permuted = labels[:]
    rng.shuffle(permuted)

    updated: list[Sample] = []
    for s, new_level in zip(samples, permuted, strict=True):
        metadata = dict(s.metadata)
        metadata["control"] = "random_label"
        metadata["original_level"] = s.level
        topic = metadata.get("topic")
        topic_suffix = f"|{topic}" if topic else ""
        sample_id = (
            f"{s.concept_name}|{metadata['split']}|{metadata['template_family']}|"
            f"{s.template_id}|{new_level}|{s.synonym}{topic_suffix}"
        )
        updated.append(
            Sample(
                sample_id=sample_id,
                concept_name=s.concept_name,
                level=new_level,
                template_id=s.template_id,
                synonym=s.synonym,
                prompt_text=s.prompt_text,
                metadata=metadata,
            )
        )
    return _attach_dataset_signature(updated)


def apply_unrelated_concept_control(
    samples: Sequence[Sample],
    unrelated_concept_name: str,
    *,
    data_spec: DataSpec | None = None,
) -> list[Sample]:
    if not samples:
        return []
    data_spec = data_spec or ProjectConfig().data
    if unrelated_concept_name not in data_spec.concepts:
        raise KeyError(f"Unknown concept: {unrelated_concept_name}")

    source_concept_name = samples[0].concept_name
    if source_concept_name not in data_spec.concepts:
        raise KeyError(f"Unknown concept: {source_concept_name}")

    source_levels = data_spec.concepts[source_concept_name].levels
    target_levels = data_spec.concepts[unrelated_concept_name].levels
    if len(source_levels) != len(target_levels):
        raise ValueError("Unrelated control requires matching number of levels")

    level_to_index = {level: i for i, level in enumerate(source_levels)}

    updated: list[Sample] = []
    for s in samples:
        idx = level_to_index[s.level]
        new_level = target_levels[idx]
        metadata = dict(s.metadata)
        metadata["control"] = "unrelated_concept"
        metadata["original_concept"] = source_concept_name
        metadata["original_level"] = s.level
        topic = metadata.get("topic")
        topic_suffix = f"|{topic}" if topic else ""
        sample_id = (
            f"{unrelated_concept_name}|{metadata['split']}|{metadata['template_family']}|"
            f"{s.template_id}|{new_level}|{s.synonym}{topic_suffix}"
        )
        updated.append(
            Sample(
                sample_id=sample_id,
                concept_name=unrelated_concept_name,
                level=new_level,
                template_id=s.template_id,
                synonym=s.synonym,
                prompt_text=s.prompt_text,
                metadata=metadata,
            )
        )
    return _attach_dataset_signature(updated)


def samples_to_dicts(samples: Sequence[Sample]) -> list[Dict[str, Any]]:
    return [asdict(s) for s in samples]
