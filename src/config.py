from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# 9 ordered sentiment levels: very negative -> very positive
SENTIMENT_LEVELS: list[str] = [
    "very_negative",
    "negative",
    "somewhat_negative",
    "slightly_negative",
    "neutral",
    "slightly_positive",
    "somewhat_positive",
    "positive",
    "very_positive",
]

# At least 10 templates with a single adjective slot.
TEMPLATES_DISCOVERY: list[str] = [
    "Overall, the experience was {w}.",
    "In summary, the product feels {w}.",
    "To be honest, the service was {w}.",
    "All in all, the event was {w}.",
    "From my view, the meal was {w}.",
]

TEMPLATES_EVAL: list[str] = [
    "Overall, the trip was {w}.",
    "In summary, the movie was {w}.",
    "To be honest, the lecture was {w}.",
    "All in all, the app is {w}.",
    "From my view, the interaction felt {w}.",
]

# Synonyms are split into disjoint discovery vs eval sets (never mix them).
# Keep 3–6 adjectives per level for each split.
SYNONYMS_DISCOVERY: dict[str, list[str]] = {
    "very_negative": ["horrible", "dreadful", "atrocious"],
    "negative": ["bad", "poor", "lousy"],
    "somewhat_negative": ["mediocre", "subpar", "lackluster"],
    "slightly_negative": ["disappointing", "underwhelming", "uneven"],
    "neutral": ["okay", "fine", "average"],
    "slightly_positive": ["decent", "pleasant", "nice"],
    "somewhat_positive": ["good", "solid", "satisfying"],
    "positive": ["great", "excellent", "wonderful"],
    "very_positive": ["amazing", "fantastic", "outstanding"],
}

SYNONYMS_EVAL: dict[str, list[str]] = {
    "very_negative": ["terrible", "awful", "abysmal"],
    "negative": ["crummy", "inferior", "weak"],
    "somewhat_negative": ["so-so", "middling", "uninspired"],
    "slightly_negative": ["meh", "shaky", "imperfect"],
    "neutral": ["fair", "standard", "typical"],
    "slightly_positive": ["agreeable", "likable", "friendly"],
    "somewhat_positive": ["worthwhile", "strong", "commendable"],
    "positive": ["brilliant", "superb", "delightful"],
    "very_positive": ["exceptional", "phenomenal", "spectacular"],
}


@dataclass(frozen=True)
class RunConfig:
    # Model / execution
    model_name: str = "gpt2"
    local_files_only: bool = True
    batch_size: int = 16
    seed: int = 0

    # Artifacts
    artifacts_dir: Path = Path("artifacts")
    plots_dir: Path = Path("plots")

    # Δ-pair construction
    # "cartesian" creates all synonym combos per adjacent level pair; "random" samples one per edge.
    delta_pair_strategy: Literal["cartesian", "random"] = "cartesian"

    # PCA / rotation
    pca_solver: Literal["full", "randomized"] = "full"
    rotation_k_mode: Literal["fixed", "min10_k90"] = "min10_k90"
    rotation_k_fixed: int = 5
    rotation_metric: Literal["mean_deg", "sum_deg"] = "mean_deg"

    # Ablation (optional)
    ablation_top_m: int = 50
    ablation_layer: int = 5  # 0-based transformer block index for neuron scoring/ablation

