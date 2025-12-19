from __future__ import annotations

import argparse
from pathlib import Path

from .config import ControlSpec, DataSpec, ExperimentConfig, load_experiment_config
from .data import generate_samples
from .capture import capture_activations, _select_adapter
from .ablation import run_ablation_pipeline
from .specificity import run_specificity
from .behavior import run_behavior, ablation_impact_on_behavior
from .geometry_runner import run_controls, run_geometry
from .utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concept geometry capture CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="Capture activations for a concept/split")
    cap.add_argument("--config", type=str, default=None, help="Path to YAML config")
    cap.add_argument("--concept", type=str, default="sentiment")
    cap.add_argument("--split", type=str, default="discovery", choices=["discovery", "eval"])
    cap.add_argument("--template_family", type=str, default="main")
    cap.add_argument("--seed", type=int, default=0)
    cap.add_argument("--model", type=str, default="distilgpt2")
    cap.add_argument("--adapter", type=str, default="gpt2")
    cap.add_argument("--batch_size", type=int, default=4)
    cap.add_argument("--use_cache", type=int, default=1)
    cap.add_argument("--control_random_labels", action="store_true")
    cap.add_argument("--control_unrelated_labels", action="store_true")
    cap.add_argument("--control_templates", action="store_true")
    cap.add_argument("--artifacts_dir", type=str, default="artifacts")
    cap.add_argument("--local_files_only", type=int, default=1)

    abl = sub.add_parser("ablate", help="Run ablation dose-response")
    abl.add_argument("--config", type=str, default=None)
    abl.add_argument("--concept", type=str, default="sentiment")
    abl.add_argument("--split", type=str, default="eval", choices=["discovery", "eval"])
    abl.add_argument("--template_family", type=str, default="main")
    abl.add_argument("--seed", type=int, default=0)
    abl.add_argument("--model", type=str, default="distilgpt2")
    abl.add_argument("--adapter", type=str, default="gpt2")
    abl.add_argument("--batch_size", type=int, default=4)
    abl.add_argument("--layer", type=int, default=0)
    abl.add_argument("--methods", type=str, default="variance,mean_abs,probe,corr")
    abl.add_argument("--m_list", type=str, default="5,10,20,40,80")
    abl.add_argument("--artifacts_dir", type=str, default="artifacts")
    abl.add_argument("--use_cache", type=int, default=1)

    spec = sub.add_parser("specificity", help="Direction similarity and transfer")
    spec.add_argument("--config", type=str, default=None)
    spec.add_argument("--concepts", type=str, default="sentiment,concreteness")
    spec.add_argument("--split", type=str, default="discovery", choices=["discovery", "eval"])
    spec.add_argument("--adapter", type=str, default="gpt2")
    spec.add_argument("--model", type=str, default="distilgpt2")
    spec.add_argument("--use_cache", type=int, default=1)
    spec.add_argument("--artifacts_dir", type=str, default="artifacts")

    beh = sub.add_parser("behavior", help="Behavioral probe and ablation impact")
    beh.add_argument("--config", type=str, default=None)
    beh.add_argument("--concept", type=str, default="sentiment")
    beh.add_argument("--adapter", type=str, default="gpt2")
    beh.add_argument("--model", type=str, default="distilgpt2")
    beh.add_argument("--layer", type=int, default=0)
    beh.add_argument("--m", type=int, default=20)
    beh.add_argument("--selection_method", type=str, default="variance")
    beh.add_argument("--use_cache", type=int, default=1)
    beh.add_argument("--artifacts_dir", type=str, default="artifacts")
    geo = sub.add_parser("geometry", help="Compute geometry metrics from cached activations (or capture if missing)")
    geo.add_argument("--config", type=str, default=None)
    geo.add_argument("--concept", type=str, default="sentiment")
    geo.add_argument("--split", type=str, default="discovery", choices=["discovery", "eval"])
    geo.add_argument("--template_family", type=str, default="main")
    geo.add_argument("--seed", type=int, default=0)
    geo.add_argument("--model", type=str, default="distilgpt2")
    geo.add_argument("--adapter", type=str, default="gpt2")
    geo.add_argument("--batch_size", type=int, default=4)
    geo.add_argument("--use_cache", type=int, default=1)
    geo.add_argument("--n_boot", type=int, default=50)
    geo.add_argument("--artifacts_dir", type=str, default="artifacts")

    ctrl = sub.add_parser("controls", help="Random label controls and permutation tests")
    ctrl.add_argument("--config", type=str, default=None)
    ctrl.add_argument("--concept", type=str, default="sentiment")
    ctrl.add_argument("--split", type=str, default="discovery", choices=["discovery", "eval"])
    ctrl.add_argument("--template_family", type=str, default="main")
    ctrl.add_argument("--seed", type=int, default=0)
    ctrl.add_argument("--model", type=str, default="distilgpt2")
    ctrl.add_argument("--adapter", type=str, default="gpt2")
    ctrl.add_argument("--batch_size", type=int, default=4)
    ctrl.add_argument("--n_shuffles", type=int, default=50)
    ctrl.add_argument("--artifacts_dir", type=str, default="artifacts")
    ctrl.add_argument("--use_cache", type=int, default=1)
    return parser.parse_args()


def cmd_capture(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config) if args.config else None
    cfg: ExperimentConfig = load_experiment_config(cfg_path)
    data_spec = DataSpec(
        concept=args.concept,
        split=args.split,  # type: ignore[arg-type]
        template_family=args.template_family,
        seed=args.seed,
        n_levels=cfg.data.n_levels,
        n_per_level=cfg.data.n_per_level,
    )
    control_spec = ControlSpec(
        random_labels=args.control_random_labels,
        unrelated_labels=args.control_unrelated_labels,
        control_templates=args.control_templates,
    )
    ensure_dir(Path(args.artifacts_dir) / "activations")
    capture_activations(
        config=cfg,
        data_spec=data_spec,
        control_spec=control_spec,
        adapter_name=args.adapter,
        model_name=args.model,
        batch_size=args.batch_size,
        artifacts_dir=Path(args.artifacts_dir),
        use_cache=bool(args.use_cache),
        local_files_only=bool(args.local_files_only),
    )


def cmd_geometry(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(Path(args.config) if args.config else None)
    data_spec = DataSpec(
        concept=args.concept,
        split=args.split,  # type: ignore[arg-type]
        template_family=args.template_family,
        seed=args.seed,
        n_levels=cfg.data.n_levels,
        n_per_level=cfg.data.n_per_level,
    )
    control_spec = ControlSpec()
    run_geometry(
        cfg=cfg,
        data_spec=data_spec,
        control_spec=control_spec,
        artifacts_dir=Path(args.artifacts_dir),
        adapter=args.adapter,
        model=args.model,
        batch_size=args.batch_size,
        use_cache=bool(args.use_cache),
        n_boot=args.n_boot,
    )


def cmd_controls(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(Path(args.config) if args.config else None)
    data_spec = DataSpec(
        concept=args.concept,
        split=args.split,  # type: ignore[arg-type]
        template_family=args.template_family,
        seed=args.seed,
        n_levels=cfg.data.n_levels,
        n_per_level=cfg.data.n_per_level,
    )
    control_spec = ControlSpec(random_labels=False)
    run_controls(
        cfg=cfg,
        data_spec=data_spec,
        control_spec=control_spec,
        artifacts_dir=Path(args.artifacts_dir),
        adapter=args.adapter,
        model=args.model,
        batch_size=args.batch_size,
        n_shuffles=args.n_shuffles,
        thresholds=[0.8, 0.9, 0.95],
        early_layers=[0, 1, 2],
        late_layers=[-3, -2, -1],
        use_cache=bool(args.use_cache),
    )


def cmd_ablate(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(Path(args.config) if args.config else None)
    data_spec = DataSpec(
        concept=args.concept,
        split=args.split,  # type: ignore[arg-type]
        template_family=args.template_family,
        seed=args.seed,
        n_levels=cfg.data.n_levels,
        n_per_level=cfg.data.n_per_level,
    )
    control_spec = ControlSpec()
    samples, ds_sig = generate_samples(cfg, data_spec=data_spec, control=control_spec)
    capture = capture_activations(
        config=cfg,
        data_spec=data_spec,
        control_spec=control_spec,
        adapter_name=args.adapter,
        model_name=args.model,
        batch_size=args.batch_size,
        artifacts_dir=Path(args.artifacts_dir),
        use_cache=bool(args.use_cache),
        local_files_only=True,
    )
    adapter = _select_adapter(args.adapter, args.model, local_files_only=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    out_dir = Path(args.artifacts_dir) / "ablation" / f"{args.concept}__{args.split}"
    run_ablation_pipeline(
        adapter=adapter,
        samples=samples,
        mlp=capture.mlp,
        residual=capture.residual,
        layer=args.layer,
        m_list=m_list,
        methods=methods,
        seed=args.seed,
        out_dir=out_dir,
    )


def cmd_specificity(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(Path(args.config) if args.config else None)
    concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    run_specificity(
        cfg=cfg,
        concepts=concepts,
        split=args.split,  # type: ignore[arg-type]
        adapter=args.adapter,
        model=args.model,
        artifacts_dir=Path(args.artifacts_dir),
        use_cache=bool(args.use_cache),
    )


def cmd_behavior(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = load_experiment_config(Path(args.config) if args.config else None)
    run_behavior(
        cfg=cfg,
        concept=args.concept,
        adapter=args.adapter,
        model=args.model,
        artifacts_dir=Path(args.artifacts_dir),
        use_cache=bool(args.use_cache),
    )
    ablation_impact_on_behavior(
        cfg=cfg,
        concept=args.concept,
        adapter_name=args.adapter,
        model=args.model,
        layer=args.layer,
        m=args.m,
        selection_method=args.selection_method,
        artifacts_dir=Path(args.artifacts_dir),
        use_cache=bool(args.use_cache),
    )


def main() -> None:
    args = parse_args()
    if args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "geometry":
        cmd_geometry(args)
    elif args.cmd == "controls":
        cmd_controls(args)
    elif args.cmd == "ablate":
        cmd_ablate(args)
    elif args.cmd == "specificity":
        cmd_specificity(args)
    elif args.cmd == "behavior":
        cmd_behavior(args)


if __name__ == "__main__":
    main()
