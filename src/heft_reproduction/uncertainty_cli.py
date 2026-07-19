"""CLI for static-HEFT runtime-uncertainty experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .scheduler import schedule_heft
from .trace_cli import DEFAULT_CONFIG, DEFAULT_INPUT
from .trace_model import build_modeled_workflow, load_trace_model_config
from .uncertainty import run_uncertainty_experiment
from .uncertainty_visualization import save_uncertainty_summary_plot
from .wfcommons import load_wfcommons_trace


DEFAULT_OUTPUT = Path("results/montage_uncertainty.json")
DEFAULT_PLOT = Path("results/montage_uncertainty_summary.png")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_cvs(value: str) -> tuple[float, ...]:
    try:
        cvs = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CV values must be comma-separated numbers") from exc
    if not cvs or any(cv < 0 or cv == float("inf") or cv != cv for cv in cvs):
        raise argparse.ArgumentTypeError("CV values must be finite and non-negative")
    return cvs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a static HEFT plan under seeded runtime uncertainty."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cvs", type=_parse_cvs, default=(0.0, 0.1, 0.3, 0.5))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"also save a summary plot (suggested: {DEFAULT_PLOT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials <= 0:
        print("Error: --trials must be positive")
        return 2
    try:
        trace = load_wfcommons_trace(args.input)
        config = load_trace_model_config(args.config)
        modeled = build_modeled_workflow(trace, config)
        baseline = schedule_heft(modeled.workflow)
        if not baseline.is_valid:
            raise ValueError(f"invalid HEFT baseline: {baseline.validation_errors}")
        experiment = run_uncertainty_experiment(
            modeled.workflow,
            baseline.schedule,
            args.cvs,
            args.trials,
            args.seed,
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 2

    payload = {
        "interpretation": {
            "study_type": "static-plan replay under simulated runtime uncertainty",
            "fixed_policy": (
                "HEFT worker assignments and planned per-worker task order are "
                "held fixed during replay"
            ),
            "uncertainty": (
                "independent task-level mean-one log-normal multipliers applied "
                "to modeled task durations"
            ),
            "perfect_information_reference": (
                "HEFT rerun after seeing sampled durations; diagnostic reference, "
                "not an optimal oracle or deployable policy"
            ),
        },
        "source": {
            "workflow": trace.name,
            "local_filename": args.input.name,
            "sha256": _sha256(args.input),
            "worker_model": config.model_name,
        },
        **experiment.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Trace: {trace.name} | Tasks: {len(trace.tasks)} | Edges: {trace.edge_count}")
    print(f"Estimated static HEFT makespan: {experiment.planned_makespan:.3f} s")
    print(f"Trials per CV: {args.trials} | Master seed: {args.seed}")
    for summary in experiment.summaries:
        print(
            f"  CV={summary.cv:.3f}: fixed plan {summary.static_plan_mean_makespan:.3f} "
            f"+/- {summary.static_plan_ci95_half_width:.3f} s | "
            f"perfect-information HEFT {summary.perfect_information_heft_mean_makespan:.3f} s | "
            f"gap {summary.mean_static_plan_gap:.3f} s"
        )
    print(f"JSON: {args.output}")

    if args.plot is not None:
        try:
            output = save_uncertainty_summary_plot(experiment, args.plot)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Summary plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
