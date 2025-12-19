from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import yaml


############################################################
# Stage 1: multi-concept configuration
############################################################


@dataclass(frozen=True)
class ConceptSpec:
    name: str
    levels: List[str]
    discovery_synonyms: Dict[str, List[str]]
    eval_synonyms: Dict[str, List[str]]
    template_families: Dict[str, Dict[str, List[str]]]  # family -> {"discovery": [...], "eval": [...]}


@dataclass(frozen=True)
class DataSpec:
    concept: str = "sentiment"
    split: Literal["discovery", "eval"] = "discovery"
    template_family: str = "main"
    seed: int = 0
    n_levels: int = 9
    n_per_level: int = 3


@dataclass(frozen=True)
class ControlSpec:
    random_labels: bool = False
    unrelated_labels: bool = False
    control_templates: bool = False


@dataclass(frozen=True)
class HashSpec:
    enable_hashing: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    concepts: Dict[str, ConceptSpec]
    data: DataSpec = field(default_factory=DataSpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    hashing: HashSpec = field(default_factory=HashSpec)

    @staticmethod
    def defaults() -> "ExperimentConfig":
        sentiment_levels = [str(i) for i in range(9)]
        concreteness_levels = [str(i) for i in range(9)]
        sentiment_templates = {
            "main": {
                "discovery": [
                    "Overall, the experience was {w}.",
                    "In summary, the product feels {w}.",
                    "To be honest, the service was {w}.",
                ],
                "eval": [
                    "Overall, the trip was {w}.",
                    "In summary, the movie was {w}.",
                    "To be honest, the lecture was {w}.",
                ],
            },
            "control": {
                "discovery": [
                    "The catalog lists a {w} item.",
                    "A standard device uses a {w} part.",
                ],
                "eval": [
                    "Each kit ships with a {w} component.",
                    "The lab relies on a {w} sample.",
                ],
            },
        }
        concreteness_templates = {
            "main": {
                "discovery": [
                    "The object on the table is a {w}.",
                    "They examined the {w} closely.",
                ],
                "eval": [
                    "She carried the {w} carefully.",
                    "The study focused on the {w}.",
                ],
            },
            "control": {
                "discovery": [
                    "The document remains {w} throughout.",
                    "The interface stays {w} in tone.",
                ],
                "eval": [
                    "The note is deliberately {w} in style.",
                    "The report stays {w} and direct.",
                ],
            },
        }
        sentiment_disc = {
            "0": ["horrible", "dreadful", "atrocious"],
            "1": ["bad", "poor", "lousy"],
            "2": ["mediocre", "subpar", "lackluster"],
            "3": ["disappointing", "underwhelming", "uneven"],
            "4": ["okay", "fine", "average"],
            "5": ["decent", "pleasant", "nice"],
            "6": ["good", "solid", "satisfying"],
            "7": ["great", "excellent", "wonderful"],
            "8": ["amazing", "fantastic", "outstanding"],
        }
        sentiment_eval = {
            "0": ["terrible", "awful", "abysmal"],
            "1": ["crummy", "inferior", "weak"],
            "2": ["so-so", "middling", "uninspired"],
            "3": ["meh", "shaky", "imperfect"],
            "4": ["fair", "standard", "typical"],
            "5": ["agreeable", "likable", "friendly"],
            "6": ["worthwhile", "strong", "commendable"],
            "7": ["brilliant", "superb", "delightful"],
            "8": ["exceptional", "phenomenal", "spectacular"],
        }
        concreteness_disc = {
            "0": ["idea", "freedom", "ethics"],
            "1": ["concept", "logic", "belief"],
            "2": ["memory", "dream", "thought"],
            "3": ["signal", "noise", "tone"],
            "4": ["stone", "metal", "branch"],
            "5": ["flower", "cup", "bottle"],
            "6": ["table", "window", "jacket"],
            "7": ["vehicle", "building", "river"],
            "8": ["mountain", "ocean", "planet"],
        }
        concreteness_eval = {
            "0": ["virtue", "myth", "justice"],
            "1": ["pattern", "value", "symbol"],
            "2": ["rumor", "music", "story"],
            "3": ["color", "echo", "pulse"],
            "4": ["rope", "grain", "leaf"],
            "5": ["pencil", "brush", "plate"],
            "6": ["laptop", "door", "blanket"],
            "7": ["bridge", "stadium", "highway"],
            "8": ["volcano", "galaxy", "continent"],
        }
        return ExperimentConfig(
            concepts={
                "sentiment": ConceptSpec(
                    name="sentiment",
                    levels=sentiment_levels,
                    discovery_synonyms=sentiment_disc,
                    eval_synonyms=sentiment_eval,
                    template_families=sentiment_templates,
                ),
                "concreteness": ConceptSpec(
                    name="concreteness",
                    levels=concreteness_levels,
                    discovery_synonyms=concreteness_disc,
                    eval_synonyms=concreteness_eval,
                    template_families=concreteness_templates,
                ),
            }
        )


def load_experiment_config(path: Optional[Path] = None) -> ExperimentConfig:
    """
    Load a YAML config if provided, otherwise return defaults.
    """
    if path is None or not path.exists():
        return ExperimentConfig.defaults()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    defaults = ExperimentConfig.defaults()
    concepts: Dict[str, ConceptSpec] = {}
    for name, spec in defaults.concepts.items():
        override = data.get("concepts", {}).get(name, {})
        concepts[name] = ConceptSpec(
            name=name,
            levels=override.get("levels", spec.levels),
            discovery_synonyms=override.get("discovery_synonyms", spec.discovery_synonyms),
            eval_synonyms=override.get("eval_synonyms", spec.eval_synonyms),
            template_families=override.get("template_families", spec.template_families),
        )
    control_obj = data.get("control", {})
    data_obj = data.get("data", {})
    hashing_obj = data.get("hashing", {})
    return ExperimentConfig(
        concepts=concepts,
        data=DataSpec(
            concept=data_obj.get("concept", defaults.data.concept),
            split=data_obj.get("split", defaults.data.split),
            template_family=data_obj.get("template_family", defaults.data.template_family),
            seed=int(data_obj.get("seed", defaults.data.seed)),
            n_levels=int(data_obj.get("n_levels", defaults.data.n_levels)),
            n_per_level=int(data_obj.get("n_per_level", defaults.data.n_per_level)),
        ),
        control=ControlSpec(
            random_labels=bool(control_obj.get("random_labels", defaults.control.random_labels)),
            unrelated_labels=bool(
                control_obj.get("unrelated_labels", defaults.control.unrelated_labels)
            ),
            control_templates=bool(
                control_obj.get("control_templates", defaults.control.control_templates)
            ),
        ),
        hashing=HashSpec(enable_hashing=bool(hashing_obj.get("enable_hashing", True))),
    )


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
    concept_mode: Literal["sentiment", "unordered"] = "sentiment"
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
