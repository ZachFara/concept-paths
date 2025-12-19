from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .config import ConceptSpec, ControlSpec, DataSpec, ExperimentConfig
from .utils import hash_samples, stable_hash_json

Split = Literal["discovery", "eval"]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    concept_name: str
    level: int
    level_label: str
    template_id: int
    template_text: str
    synonym: str
    prompt_text: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DeltaPair:
    """
    Stub for compatibility with downstream modules; not used in Stage 1.
    """

    split: Split
    template_id: int
    neg_level_id: int
    pos_level_id: int
    neg_word: str
    pos_word: str
    neg_key: str
    pos_key: str


def _assert_disjoint(a: List[str], b: List[str], label: str) -> None:
    overlap = set(a).intersection(set(b))
    if overlap:
        raise ValueError(f"Leakage detected: {label} overlap: {sorted(overlap)[:5]}")


def _pick_synonyms(synonyms: Dict[str, List[str]], n_per_level: int, rng: np.random.Generator) -> Dict[str, List[str]]:
    picked: Dict[str, List[str]] = {}
    for lvl, words in synonyms.items():
        if not words:
            picked[lvl] = []
            continue
        words_shuffled = list(words)
        rng.shuffle(words_shuffled)
        picked[lvl] = words_shuffled[: min(n_per_level, len(words_shuffled))]
    return picked


def shuffle_labels_preserve_counts(levels: List[int], rng: np.random.Generator) -> List[int]:
    """
    Shuffle labels while preserving per-level counts.
    """
    arr = np.array(levels, dtype=np.int64)
    rng.shuffle(arr)
    return arr.tolist()


def _assign_unrelated_labels(levels: List[int], n_levels: int, rng: np.random.Generator) -> List[int]:
    counts = np.bincount(levels, minlength=n_levels)
    expanded: List[int] = []
    for lvl, c in enumerate(counts):
        expanded.extend([lvl] * int(c))
    rng.shuffle(expanded)
    return expanded


def _select_templates(spec: ConceptSpec, split: Split, family: str) -> List[str]:
    if family not in spec.template_families:
        raise ValueError(f"Template family '{family}' not found for concept {spec.name}")
    templates = spec.template_families[family].get(split)
    if templates is None:
        raise ValueError(f"Split '{split}' missing templates for family '{family}' in concept {spec.name}")
    return list(templates)


def _build_sample_id(concept: str, split: Split, template_id: int, level: int, idx: int) -> str:
    return f"{concept}-{split}-t{template_id}-l{level}-i{idx}"


def generate_samples(
    config: ExperimentConfig,
    data_spec: Optional[DataSpec] = None,
    control: Optional[ControlSpec] = None,
) -> Tuple[List[Sample], str]:
    """
    Generate samples for a concept with optional controls.
    Returns (samples, dataset_signature).
    """
    data_spec = data_spec or config.data
    control = control or config.control
    rng = np.random.default_rng(data_spec.seed)

    if data_spec.concept not in config.concepts:
        raise ValueError(f"Unknown concept: {data_spec.concept}")
    concept_spec = config.concepts[data_spec.concept]
    levels = concept_spec.levels[: data_spec.n_levels]

    # disjoint synonym guard
    for lvl in levels:
        _assert_disjoint(
            concept_spec.discovery_synonyms.get(lvl, []),
            concept_spec.eval_synonyms.get(lvl, []),
            f"synonyms level {lvl}",
        )

    split: Split = data_spec.split
    base_family = data_spec.template_family
    family = "control" if control.control_templates else base_family
    templates = _select_templates(concept_spec, split, family)
    # template disjointness guard
    other_split = "eval" if split == "discovery" else "discovery"
    other_templates = _select_templates(concept_spec, other_split, family)
    _assert_disjoint(list(templates), list(other_templates), f"templates_{family}")

    syn_source = (
        concept_spec.discovery_synonyms if split == "discovery" else concept_spec.eval_synonyms
    )
    syn_selected = _pick_synonyms(syn_source, data_spec.n_per_level, rng)

    samples: List[Sample] = []
    for template_id, template_text in enumerate(templates):
        for level_idx, level_label in enumerate(levels):
            words = syn_selected.get(level_label, [])
            for i, synonym in enumerate(words):
                prompt_text = template_text.format(w=synonym)
                sample_id = _build_sample_id(concept_spec.name, split, template_id, level_idx, i)
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        concept_name=concept_spec.name,
                        level=level_idx,
                        level_label=level_label,
                        template_id=template_id,
                        template_text=template_text,
                        synonym=synonym,
                        prompt_text=prompt_text,
                        metadata={
                            "split": split,
                            "template_family": family,
                            "original_level": level_idx,
                            "control_random": control.random_labels,
                            "control_unrelated": control.unrelated_labels,
                            "control_templates": control.control_templates,
                        },
                    )
                )

    if control.random_labels and not control.unrelated_labels:
        shuffled = shuffle_labels_preserve_counts([s.level for s in samples], rng)
        for s, new_level in zip(samples, shuffled, strict=True):
            s.metadata["original_level"] = s.level
            object.__setattr__(s, "level", int(new_level))
            object.__setattr__(s, "level_label", levels[int(new_level)])
    if control.unrelated_labels:
        unrelated = _assign_unrelated_labels([s.level for s in samples], len(levels), rng)
        for s, new_level in zip(samples, unrelated, strict=True):
            s.metadata["original_level"] = s.level
            object.__setattr__(s, "level", int(new_level))
            object.__setattr__(s, "level_label", levels[int(new_level)])

    dataset_signature = hash_samples(samples)
    for s in samples:
        s.metadata["dataset_signature"] = dataset_signature
    return samples, dataset_signature
