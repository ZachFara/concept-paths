from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .config import StimuliConfig
from .io import hash_prompts
from .utils import set_seed

Split = Literal["discovery", "eval"]


@dataclass(frozen=True)
class StimulusSample:
    split: Split
    family: str
    template_id: int
    template: str
    level_id: int
    level: str
    word: str

    @property
    def prompt(self) -> str:
        return self.template.format(w=self.word)

    @property
    def key(self) -> str:
        return f"{self.split}|{self.family}|{self.template_id}|{self.level_id}|{self.word}"

    @property
    def ordinal_index(self) -> int:
        return self.level_id


@dataclass(frozen=True)
class DeltaPair:
    split: Split
    family: str
    template_id: int
    neg_level_id: int
    pos_level_id: int
    neg_word: str
    pos_word: str
    neg_key: str
    pos_key: str


def _permute_levels(levels: List[str], rng: np.random.Generator) -> List[str]:
    levels_copy = levels.copy()
    rng.shuffle(levels_copy)
    return levels_copy


def _choose_synonyms(
    synonyms: Dict[str, List[str]],
    permute: bool,
    levels: List[str],
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    if not permute:
        return synonyms
    # Permute mapping of level -> synonyms (shuffled assignment)
    shuffled_levels = _permute_levels(levels, rng)
    return {lvl: synonyms[src] for lvl, src in zip(levels, shuffled_levels, strict=True)}


def _assert_disjoint(a: Iterable[str], b: Iterable[str], label: str) -> None:
    set_a = set(a)
    set_b = set(b)
    overlap = set_a.intersection(set_b)
    if overlap:
        raise ValueError(f"Leakage detected: {label} overlap: {sorted(overlap)[:5]}")


def generate_stimuli(
    cfg: StimuliConfig,
    *,
    split: Split,
    permute_labels: bool = False,
    concept_mode: Literal["sentiment", "unordered"] = "sentiment",
    seed: int = 0,
) -> List[StimulusSample]:
    rng = np.random.default_rng(seed)
    samples: List[StimulusSample] = []
    if concept_mode == "unordered":
        levels = list(cfg.unordered_categories.keys())
        synonyms_discovery = cfg.unordered_categories
        synonyms_eval = cfg.unordered_categories
    else:
        levels = list(cfg.sentiment_levels)
        synonyms_discovery = cfg.synonyms_discovery
        synonyms_eval = cfg.synonyms_eval
        # Leakage guard: disjoint synonym sets between splits
        for lvl in levels:
            _assert_disjoint(
                synonyms_discovery.get(lvl, []),
                synonyms_eval.get(lvl, []),
                f"synonyms level {lvl}",
            )

    for family, splits in cfg.families.items():
        templates = splits["discovery"] if split == "discovery" else splits["eval"]
        syn = synonyms_discovery if split == "discovery" else synonyms_eval
        syn = _choose_synonyms(syn, permute_labels, levels, rng)
        for template_id, template in enumerate(templates):
            for level_id, level in enumerate(levels):
                for word in syn[level]:
                    samples.append(
                        StimulusSample(
                            split=split,
                            family=family,
                            template_id=template_id,
                            template=template,
                            level_id=level_id,
                            level=level,
                            word=word,
                        )
                    )
    return samples


def assert_disjoint_prompts(discovery_prompts: Sequence[str], eval_prompts: Sequence[str]) -> None:
    _assert_disjoint(discovery_prompts, eval_prompts, "prompt")


def prompt_hash_manifest(discovery_prompts: Sequence[str], eval_prompts: Sequence[str]) -> dict:
    d_hashes = hash_prompts(discovery_prompts)
    e_hashes = hash_prompts(eval_prompts)
    assert_disjoint_prompts(d_hashes, e_hashes)
    return {"discovery": d_hashes, "eval": e_hashes}


def build_delta_pairs(
    samples: List[StimulusSample],
    *,
    strategy: Literal["cartesian", "random"] = "cartesian",
    seed: int = 0,
) -> List[DeltaPair]:
    if not samples:
        return []
    split: Split = samples[0].split
    rng = np.random.default_rng(seed)
    pairs: List[DeltaPair] = []

    by_family: Dict[str, List[StimulusSample]] = {}
    for s in samples:
        by_family.setdefault(s.family, []).append(s)

    for family, fam_samples in by_family.items():
        by_template: Dict[int, List[StimulusSample]] = {}
        for s in fam_samples:
            by_template.setdefault(s.template_id, []).append(s)
        for template_id, temp_samples in by_template.items():
            temp_samples_sorted = sorted(temp_samples, key=lambda x: x.level_id)
            max_level = max(s.level_id for s in temp_samples_sorted)
            # Group words per level
            words_by_level: Dict[int, List[str]] = {}
            for s in temp_samples_sorted:
                words_by_level.setdefault(s.level_id, []).append(s.word)
            for neg_level_id in range(max_level):
                pos_level_id = neg_level_id + 1
                neg_words = words_by_level[neg_level_id]
                pos_words = words_by_level[pos_level_id]
                neg_choices = neg_words
                pos_choices = pos_words
                if strategy == "random":
                    neg_choices = [neg_words[int(rng.integers(0, len(neg_words)))]]
                    pos_choices = [pos_words[int(rng.integers(0, len(pos_words)))]]
                for nw in neg_choices:
                    for pw in pos_choices:
                        neg_key = f"{split}|{family}|{template_id}|{neg_level_id}|{nw}"
                        pos_key = f"{split}|{family}|{template_id}|{pos_level_id}|{pw}"
                        pairs.append(
                            DeltaPair(
                                split=split,
                                family=family,
                                template_id=template_id,
                                neg_level_id=neg_level_id,
                                pos_level_id=pos_level_id,
                                neg_word=nw,
                                pos_word=pw,
                                neg_key=neg_key,
                                pos_key=pos_key,
                            )
                        )
    return pairs


def subsample_pairs(pairs: List[DeltaPair], frac: Optional[float], seed: int) -> List[DeltaPair]:
    if frac is None or frac >= 1.0:
        return pairs
    rng = np.random.default_rng(seed)
    k = max(1, int(len(pairs) * frac))
    idx = rng.choice(len(pairs), size=k, replace=False)
    return [pairs[i] for i in idx]
