from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import yaml


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

TOPIC_SWAP_TEMPLATES_DISCOVERY: list[str] = [
    "Overall, the {topic} was {w}.",
    "In summary, the {topic} feels {w}.",
    "To be honest, the {topic} was {w}.",
    "All in all, the {topic} was {w}.",
    "From my view, the {topic} was {w}.",
]

TOPIC_SWAP_TEMPLATES_EVAL: list[str] = [
    "In retrospect, the {topic} seemed {w}.",
    "From my perspective, the {topic} felt {w}.",
    "Net-net, the {topic} proved {w}.",
    "By the end, the {topic} was {w}.",
    "If I'm honest, the {topic} came across {w}.",
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

# 9 ordered concreteness levels: very abstract -> very concrete
CONCRETENESS_LEVELS: list[str] = [
    "very_abstract",
    "abstract",
    "somewhat_abstract",
    "slightly_abstract",
    "neutral",
    "slightly_concrete",
    "somewhat_concrete",
    "concrete",
    "very_concrete",
]

CONCRETENESS_TEMPLATES_DISCOVERY: list[str] = [
    "The object seemed {w}.",
    "It was a {w} example.",
    "The description felt {w}.",
    "In practice, it was {w}.",
    "The model remained {w}.",
]

CONCRETENESS_TEMPLATES_EVAL: list[str] = [
    "The artifact looked {w}.",
    "The case was {w}.",
    "Overall, the instance was {w}.",
    "In the end, it proved {w}.",
    "The explanation stayed {w}.",
]

CONCRETENESS_SYNONYMS_DISCOVERY: dict[str, list[str]] = {
    "very_abstract": ["conceptual", "theoretical", "notional"],
    "abstract": ["immaterial", "nonphysical", "symbolic"],
    "somewhat_abstract": ["figurative", "generalized", "broad"],
    "slightly_abstract": ["vague", "hazy", "loose"],
    "neutral": ["ordinary", "standard", "typical"],
    "slightly_concrete": ["specific", "particular", "defined"],
    "somewhat_concrete": ["detailed", "explicit", "clear"],
    "concrete": ["tangible", "material", "physical"],
    "very_concrete": ["palpable", "solid", "touchable"],
}

CONCRETENESS_SYNONYMS_EVAL: dict[str, list[str]] = {
    "very_abstract": ["intangible", "esoteric", "metaphysical"],
    "abstract": ["incorporeal", "mental", "cerebral"],
    "somewhat_abstract": ["diffuse", "approximate", "generic"],
    "slightly_abstract": ["blurred", "indistinct", "ambiguous"],
    "neutral": ["common", "regular", "routine"],
    "slightly_concrete": ["distinct", "precise", "exact"],
    "somewhat_concrete": ["definite", "crisp", "direct"],
    "concrete": ["real", "substantial", "corporeal"],
    "very_concrete": ["graspable", "observable", "hands-on"],
}

CONTROL_TEMPLATES_DISCOVERY: list[str] = [
    "The word was {w}.",
    "The label read {w}.",
    "The token looked like {w}.",
    "The term is {w}.",
    "It was marked {w}.",
]

CONTROL_TEMPLATES_EVAL: list[str] = [
    "The phrase was {w}.",
    "The tag said {w}.",
    "The entry read {w}.",
    "The item was {w}.",
    "It was labeled {w}.",
]

TOPIC_SWAP_TOPICS_DISCOVERY: list[str] = [
    "trip",
    "meal",
    "lecture",
    "app",
    "service",
]

TOPIC_SWAP_TOPICS_EVAL: list[str] = [
    "movie",
    "product",
    "event",
    "interaction",
    "interface",
]


@dataclass(frozen=True)
class StimuliFamily:
    name: str
    templates_discovery: List[str]
    templates_eval: List[str]


@dataclass(frozen=True)
class StimuliConfig:
    sentiment_levels: List[str] = field(default_factory=lambda: SENTIMENT_LEVELS)
    synonyms_discovery: Dict[str, List[str]] = field(default_factory=lambda: SYNONYMS_DISCOVERY)
    synonyms_eval: Dict[str, List[str]] = field(default_factory=lambda: SYNONYMS_EVAL)
    families: Dict[str, Dict[str, List[str]]] = field(
        default_factory=lambda: {
            "adjective_clause": {
                "discovery": TEMPLATES_DISCOVERY,
                "eval": TEMPLATES_EVAL,
            },
            # Minimal orthogonal additions for varied structures.
            "verb_predicate": {
                "discovery": [
                    "The service felt {w}.",
                    "The interface seems {w}.",
                ],
                "eval": [
                    "The presentation sounded {w}.",
                    "The response was {w}.",
                ],
            },
            "delayed_sentiment": {
                "discovery": [
                    "What stood out most was that it was {w}.",
                    "To put it simply, it ended up {w}.",
                ],
                "eval": [
                    "After everything, it turned out {w}.",
                    "In the end, the outcome was {w}.",
                ],
            },
            "negation": {
                "discovery": [
                    "It was not entirely {w}.",
                    "The mood wasn’t exactly {w}.",
                ],
                "eval": [
                    "Overall it was nowhere near {w}.",
                    "Frankly, it was anything but {w}.",
                ],
            },
        }
    )
    unordered_categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "cat_a": ["alpha", "bravo", "charlie"],
            "cat_b": ["delta", "echo", "foxtrot"],
            "cat_c": ["golf", "hotel", "india"],
            "cat_d": ["juliet", "kilo", "lima"],
            "cat_e": ["mike", "november", "oscar"],
        }
    )


@dataclass(frozen=True)
class GeometryConfig:
    delta_pair_strategy: Literal["cartesian", "random"] = "cartesian"
    pca_solver: Literal["full", "randomized"] = "full"
    rotation_k_mode: Literal["fixed", "min10_k90"] = "min10_k90"
    rotation_k_fixed: int = 5
    rotation_metric: Literal["mean_deg", "sum_deg"] = "mean_deg"
    permute_labels: bool = False
    concept_mode: Literal["sentiment", "unordered", "topic_control"] = "sentiment"
    pair_subsample_frac: Optional[float] = None
    random_baseline_directions: int = 10
    random_baseline_subspaces: int = 5
    pc1_anchor: bool = True


@dataclass(frozen=True)
class AblationConfig:
    top_m: int = 50
    selectors: List[str] = field(
        default_factory=lambda: ["lookahead", "local_corr", "local_ridge"]
    )


@dataclass(frozen=True)
class RunConfig:
    model_name: str = "gpt2"
    local_files_only: bool = True
    batch_size: int = 16
    seeds: List[int] = field(default_factory=lambda: [0])
    device: Optional[str] = None  # override auto-detect
    artifacts_dir: Path = Path("artifacts")
    plots_dir: Path = Path("plots")
    run_id: Optional[str] = None
    stimuli: StimuliConfig = field(default_factory=StimuliConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @staticmethod
    def from_dict(obj: dict) -> "RunConfig":
        """
        Construct a RunConfig from a plain dictionary (e.g., loaded from YAML/JSON).
        """
        stimuli_obj = obj.get("stimuli", {})
        geometry_obj = obj.get("geometry", {})
        ablation_obj = obj.get("ablation", {})
        return RunConfig(
            model_name=obj.get("model_name", "gpt2"),
            local_files_only=bool(obj.get("local_files_only", True)),
            batch_size=int(obj.get("batch_size", 16)),
            seeds=list(obj.get("seeds", [0])),
            device=obj.get("device", None),
            artifacts_dir=Path(obj.get("artifacts_dir", "artifacts")),
            plots_dir=Path(obj.get("plots_dir", "plots")),
            run_id=obj.get("run_id", None),
            stimuli=StimuliConfig(
                sentiment_levels=list(stimuli_obj.get("sentiment_levels", SENTIMENT_LEVELS)),
                synonyms_discovery=stimuli_obj.get("synonyms_discovery", SYNONYMS_DISCOVERY),
                synonyms_eval=stimuli_obj.get("synonyms_eval", SYNONYMS_EVAL),
                families=stimuli_obj.get("families", None)
                or StimuliConfig().families,
                unordered_categories=stimuli_obj.get(
                    "unordered_categories", StimuliConfig().unordered_categories
                ),
            ),
            geometry=GeometryConfig(
                delta_pair_strategy=geometry_obj.get(
                    "delta_pair_strategy", GeometryConfig().delta_pair_strategy
                ),
                pca_solver=geometry_obj.get("pca_solver", GeometryConfig().pca_solver),
                rotation_k_mode=geometry_obj.get(
                    "rotation_k_mode", GeometryConfig().rotation_k_mode
                ),
                rotation_k_fixed=int(
                    geometry_obj.get("rotation_k_fixed", GeometryConfig().rotation_k_fixed)
                ),
                rotation_metric=geometry_obj.get(
                    "rotation_metric", GeometryConfig().rotation_metric
                ),
                permute_labels=bool(
                    geometry_obj.get("permute_labels", GeometryConfig().permute_labels)
                ),
                concept_mode=geometry_obj.get("concept_mode", GeometryConfig().concept_mode),
                pair_subsample_frac=geometry_obj.get(
                    "pair_subsample_frac", GeometryConfig().pair_subsample_frac
                ),
                random_baseline_directions=int(
                    geometry_obj.get(
                        "random_baseline_directions",
                        GeometryConfig().random_baseline_directions,
                    )
                ),
                random_baseline_subspaces=int(
                    geometry_obj.get(
                        "random_baseline_subspaces",
                        GeometryConfig().random_baseline_subspaces,
                    )
                ),
                pc1_anchor=bool(geometry_obj.get("pc1_anchor", GeometryConfig().pc1_anchor)),
            ),
            ablation=AblationConfig(
                top_m=int(ablation_obj.get("top_m", AblationConfig().top_m)),
                selectors=list(ablation_obj.get("selectors", AblationConfig().selectors)),
            ),
        )


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    levels: List[str]
    synonyms: Dict[str, Dict[str, List[str]]]
    templates: Dict[str, Dict[str, List[str]]]

    @staticmethod
    def from_dict(
        name: str, obj: dict, defaults: Optional["ConceptSpec"] = None
    ) -> "ConceptSpec":
        if defaults is None:
            defaults = ConceptSpec(name=name, levels=[], synonyms={}, templates={})
        return ConceptSpec(
            name=name,
            levels=list(obj.get("levels", defaults.levels)),
            synonyms=dict(obj.get("synonyms", defaults.synonyms)),
            templates=dict(obj.get("templates", defaults.templates)),
        )


def _default_concepts() -> Dict[str, ConceptSpec]:
    return {
        "sentiment": ConceptSpec(
            name="sentiment",
            levels=SENTIMENT_LEVELS,
            synonyms={
                "discovery": SYNONYMS_DISCOVERY,
                "eval": SYNONYMS_EVAL,
            },
            templates={
                "adjective_clause": {
                    "discovery": TEMPLATES_DISCOVERY,
                    "eval": TEMPLATES_EVAL,
                },
                "topic_swap_fixed_sentiment": {
                    "discovery": TOPIC_SWAP_TEMPLATES_DISCOVERY,
                    "eval": TOPIC_SWAP_TEMPLATES_EVAL,
                },
            },
        ),
        "concreteness": ConceptSpec(
            name="concreteness",
            levels=CONCRETENESS_LEVELS,
            synonyms={
                "discovery": CONCRETENESS_SYNONYMS_DISCOVERY,
                "eval": CONCRETENESS_SYNONYMS_EVAL,
            },
            templates={
                "descriptor_clause": {
                    "discovery": CONCRETENESS_TEMPLATES_DISCOVERY,
                    "eval": CONCRETENESS_TEMPLATES_EVAL,
                }
            },
        ),
    }


@dataclass(frozen=True)
class DataSpec:
    concepts: Dict[str, ConceptSpec] = field(default_factory=_default_concepts)
    split_names: List[str] = field(default_factory=lambda: ["discovery", "eval"])

    @staticmethod
    def from_dict(obj: dict) -> "DataSpec":
        defaults = _default_concepts()
        concepts_obj = obj.get("concepts", None)
        if concepts_obj:
            concepts = {}
            for name, spec in concepts_obj.items():
                concepts[name] = ConceptSpec.from_dict(name, spec, defaults.get(name))
        else:
            concepts = defaults
        return DataSpec(
            concepts=concepts,
            split_names=list(obj.get("split_names", ["discovery", "eval"])),
        )


@dataclass(frozen=True)
class ControlSpec:
    neutral_template_family: str = "neutral"
    neutral_templates: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "discovery": CONTROL_TEMPLATES_DISCOVERY,
            "eval": CONTROL_TEMPLATES_EVAL,
        }
    )
    topic_swap_topics: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "discovery": TOPIC_SWAP_TOPICS_DISCOVERY,
            "eval": TOPIC_SWAP_TOPICS_EVAL,
        }
    )

    @staticmethod
    def from_dict(obj: dict) -> "ControlSpec":
        return ControlSpec(
            neutral_template_family=obj.get("neutral_template_family", "neutral"),
            neutral_templates=obj.get(
                "neutral_templates",
                {"discovery": CONTROL_TEMPLATES_DISCOVERY, "eval": CONTROL_TEMPLATES_EVAL},
            ),
            topic_swap_topics=obj.get(
                "topic_swap_topics",
                {"discovery": TOPIC_SWAP_TOPICS_DISCOVERY, "eval": TOPIC_SWAP_TOPICS_EVAL},
            ),
        )


@dataclass(frozen=True)
class HashSpec:
    algorithm: Literal["sha256"] = "sha256"

    @staticmethod
    def from_dict(obj: dict) -> "HashSpec":
        return HashSpec(algorithm=obj.get("algorithm", "sha256"))


@dataclass(frozen=True)
class ModelSpec:
    backend: Literal["nnsight", "hooks"] = "nnsight"
    capture_site: str = "block_out"
    pooling: str = "last"

    @staticmethod
    def from_dict(obj: dict) -> "ModelSpec":
        return ModelSpec(
            backend=obj.get("backend", "nnsight"),
            capture_site=obj.get("capture_site", "block_out"),
            pooling=obj.get("pooling", "last"),
        )


@dataclass(frozen=True)
class ProjectConfig:
    data: DataSpec = field(default_factory=DataSpec)
    controls: ControlSpec = field(default_factory=ControlSpec)
    hashing: HashSpec = field(default_factory=HashSpec)
    model: ModelSpec = field(default_factory=ModelSpec)

    @staticmethod
    def from_dict(obj: dict) -> "ProjectConfig":
        return ProjectConfig(
            data=DataSpec.from_dict(obj.get("data", {})),
            controls=ControlSpec.from_dict(obj.get("controls", {})),
            hashing=HashSpec.from_dict(obj.get("hashing", {})),
            model=ModelSpec.from_dict(obj.get("model", {})),
        )


def load_config(path: Optional[Path] = None) -> ProjectConfig:
    if path is None or not path.exists():
        return ProjectConfig()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return ProjectConfig.from_dict(data)
