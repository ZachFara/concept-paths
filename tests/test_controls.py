import numpy as np

from src.config import StimuliConfig
from src.stimuli import generate_stimuli


def test_permutation_changes_mapping():
    cfg = StimuliConfig()
    base = generate_stimuli(cfg, split="discovery", permute_labels=False, concept_mode="sentiment", seed=0)
    perm = generate_stimuli(cfg, split="discovery", permute_labels=True, concept_mode="sentiment", seed=0)
    base_words = {(s.level, s.word) for s in base}
    perm_words = {(s.level, s.word) for s in perm}
    # Permutation keeps same word set but reassigns levels; intersection of level/word pairs should shrink
    assert len(base_words.intersection(perm_words)) < len(base_words)
