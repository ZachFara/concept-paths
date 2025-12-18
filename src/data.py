from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import (
    SENTIMENT_LEVELS,
    SYNONYMS_DISCOVERY,
    SYNONYMS_EVAL,
    TEMPLATES_DISCOVERY,
    TEMPLATES_EVAL,
)


Split = Literal["discovery", "eval"]


@dataclass(frozen=True)
class Sample:
    split: Split
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
        # Unique within a split due to disjoint templates/synonyms across splits.
        return f"{self.template_id}|{self.level_id}|{self.word}"


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


def get_templates(split: Split) -> list[str]:
    return TEMPLATES_DISCOVERY if split == "discovery" else TEMPLATES_EVAL


def get_synonyms(split: Split) -> dict[str, list[str]]:
    return SYNONYMS_DISCOVERY if split == "discovery" else SYNONYMS_EVAL


def generate_samples(split: Split) -> list[Sample]:
    templates = get_templates(split)
    synonyms = get_synonyms(split)

    samples: list[Sample] = []
    for template_id, template in enumerate(templates):
        for level_id, level in enumerate(SENTIMENT_LEVELS):
            for word in synonyms[level]:
                samples.append(
                    Sample(
                        split=split,
                        template_id=template_id,
                        template=template,
                        level_id=level_id,
                        level=level,
                        word=word,
                    )
                )
    return samples


def index_samples(samples: list[Sample]) -> dict[str, int]:
    return {s.key: i for i, s in enumerate(samples)}


def build_delta_pairs(
    samples: list[Sample],
    *,
    strategy: Literal["cartesian", "random"] = "cartesian",
    seed: int = 0,
) -> list[DeltaPair]:
    """
    Build "small step" adjacent sentiment Δ pairs:
    (level i -> level i+1), same template, optionally varying synonyms.

    strategy:
      - "cartesian": all synonym combinations for each adjacent edge
      - "random": one sampled synonym pair per edge (deterministic via seed)
    """
    if not samples:
        return []

    split: Split = samples[0].split
    templates = get_templates(split)
    synonyms = get_synonyms(split)
    sample_index = index_samples(samples)

    rng = np.random.default_rng(seed)
    pairs: list[DeltaPair] = []

    for template_id, _template in enumerate(templates):
        for neg_level_id in range(len(SENTIMENT_LEVELS) - 1):
            pos_level_id = neg_level_id + 1
            neg_level = SENTIMENT_LEVELS[neg_level_id]
            pos_level = SENTIMENT_LEVELS[pos_level_id]

            neg_words = synonyms[neg_level]
            pos_words = synonyms[pos_level]

            if strategy == "random":
                neg_words = [neg_words[int(rng.integers(0, len(neg_words)))]]
                pos_words = [pos_words[int(rng.integers(0, len(pos_words)))]]

            for neg_word in neg_words:
                for pos_word in pos_words:
                    neg_key = f"{template_id}|{neg_level_id}|{neg_word}"
                    pos_key = f"{template_id}|{pos_level_id}|{pos_word}"
                    # Because samples are a full Cartesian product, these should always exist.
                    if neg_key not in sample_index or pos_key not in sample_index:
                        continue
                    pairs.append(
                        DeltaPair(
                            split=split,
                            template_id=template_id,
                            neg_level_id=neg_level_id,
                            pos_level_id=pos_level_id,
                            neg_word=neg_word,
                            pos_word=pos_word,
                            neg_key=neg_key,
                            pos_key=pos_key,
                        )
                    )

    return pairs

