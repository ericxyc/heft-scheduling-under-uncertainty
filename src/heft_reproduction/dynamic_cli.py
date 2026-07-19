"""CLI for dynamic multi-workflow scheduling comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Sequence

from .dynamic_models import POLICY_NAMES
from .dynamic_scenario import build_dynamic_scenario, template_from_modeled
from .dynamic_simulator import simulate_dynamic_scenario
from .dynamic_visualization import save_dynamic_comparison_plot
from .trace_cli import DEFAULT_CONFIG, DEFAULT_INPUT
from .trace_model import build_modeled_workflow, load_trace_model_config
from .wfcommons import load_wfcommons_trace


DEFAULT_OUTPUT = Path("results/dynamic_workflows.json")
DEFAULT_PLOT = Path("results/dynamic_workflows_comparison.png")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_policies(value: str) -> tuple[str, ...]:
    policies = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(policies) - set(POLICY_NAMES))
    if not policies:
        raise argparse.ArgumentTypeError("at least one policy is required")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown policies {unknown}; choices are {list(POLICY_NAMES)}"
        )
    if len(set(policies)) != len(policies):
        raise argparse.ArgumentTypeError("policy names must be unique")
    return policies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare online policies on dynamically arriving workflow DAGs."
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        default=None,
        help=(
            "WfFormat JSON path; repeat for multiple templates "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workflows", type=int, default=12)
    parser.add_argument("--mean-interarrival", type=float, default=45.0)
    parser.add_argument("--cv", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--aging-weight", type=float, default=1.0)
    parser.add_argument(
        "--policies",
        type=_parse_policies,
        default=POLICY_NAMES,
        help="comma-separated policy names",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"also save a comparison plot (suggested: {DEFAULT_PLOT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workflows <= 0:
        print("Error: --workflows must be positive")
        return 2
    if not isfinite(args.mean_interarrival) or args.mean_interarrival < 0:
        print("Error: --mean-interarrival must be finite and non-negative")
        return 2
    if not isfinite(args.cv) or args.cv < 0:
        print("Error: --cv must be finite and non-negative")
        return 2
    if not isfinite(args.aging_weight) or args.aging_weight < 0:
        print("Error: --aging-weight must be finite and non-negative")
        return 2

    input_paths = args.inputs or [DEFAULT_INPUT]
    try:
        config = load_trace_model_config(args.config)
        modeled_workflows = [
            build_modeled_workflow(
                load_wfcommons_trace(path),
                config,
            )
            for path in input_paths
        ]
        templates = [
            template_from_modeled(
                modeled,
                name=(
                    modeled.trace.source_path.stem
                    if len(input_paths) == 1
                    else f"{index + 1:02d}-{modeled.trace.source_path.stem}"
                ),
            )
            for index, modeled in enumerate(modeled_workflows)
        ]
        scenario = build_dynamic_scenario(
            templates=templates,
            workflow_count=args.workflows,
            mean_interarrival_time=args.mean_interarrival,
            runtime_cv=args.cv,
            seed=args.seed,
        )
        results = [
            simulate_dynamic_scenario(
                scenario,
                policy,
                aging_weight=args.aging_weight,
            )
            for policy in args.policies
        ]
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    payload = {
        "interpretation": {
            "study_type": (
                "trace-driven dynamic multi-workflow scheduling simulation"
            ),
            "arrival_semantics": (
                "complete workflow DAGs arrive over time; future workflows "
                "are hidden until arrival"
            ),
            "execution_semantics": (
                "non-preemptive tasks, fixed simulated workers, dependency "
                "and cross-worker communication constraints"
            ),
            "information_boundary": (
                "policies use estimated durations and do not observe future "
                "arrivals or uncompleted-task realized durations"
            ),
            "scheduler_wall_time_note": (
                "wall-clock policy timing is machine-dependent; candidate "
                "evaluations are the deterministic overhead proxy"
            ),
        },
        "sources": [
            {
                "template": template.name,
                "local_filename": path.name,
                "sha256": _sha256(path),
                "schema_version": modeled.trace.schema_version,
                "task_count": len(template.workflow.tasks),
                "edge_count": len(template.workflow.communication_costs),
            }
            for path, modeled, template in zip(
                input_paths,
                modeled_workflows,
                templates,
            )
        ],
        "worker_model": config.to_dict(),
        "policy_parameters": {"aging_weight": args.aging_weight},
        "scenario": scenario.to_dict(),
        "policy_results": [result.to_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Dynamic scenario: {len(scenario.instances)} workflows | "
        f"{scenario.task_count} tasks | CV={scenario.runtime_cv:.3f} | "
        f"mean inter-arrival={scenario.mean_interarrival_time:.3f} s"
    )
    for result in results:
        metrics = result.metrics
        print(
            f"  {result.policy}: mean JCT {metrics.mean_jct:.3f} s | "
            f"P95 JCT {metrics.p95_jct:.3f} s | "
            f"mean wait {metrics.mean_task_queue_wait:.3f} s | "
            f"utilization {metrics.average_utilization:.1%} | "
            f"throughput {metrics.throughput_workflows_per_second:.5f}/s"
        )
        print(
            f"    decisions {metrics.committed_decisions} | "
            f"candidate evaluations {metrics.candidate_evaluations} | "
            f"scheduler wall {metrics.scheduler_wall_seconds:.6f} s | "
            f"valid {result.is_valid}"
        )
    print(f"JSON: {args.output}")

    if args.plot is not None:
        try:
            output = save_dynamic_comparison_plot(results, args.plot)
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Comparison plot: {output}")

    return 0 if all(result.is_valid for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
