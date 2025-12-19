from __future__ import annotations

import argparse
from pathlib import Path

from .config import ControlSpec, DataSpec, ExperimentConfig, load_experiment_config
from .capture import capture_activations
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


def main() -> None:
    args = parse_args()
    if args.cmd == "capture":
        cmd_capture(args)


if __name__ == "__main__":
    main()
