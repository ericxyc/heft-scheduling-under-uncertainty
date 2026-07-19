"""CLI for the Phase 4B multi-family dynamic scheduling benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .benchmark_data import load_benchmark_corpus
from .benchmark_visualization import save_benchmark_plot
from .dynamic_benchmark import run_dynamic_benchmark
from .dynamic_cli import _parse_policies
from .dynamic_models import POLICY_NAMES
from .trace_cli import DEFAULT_CONFIG
from .trace_model import load_trace_model_config


DEFAULT_MANIFEST = Path("configs/workflow_benchmark.json")
DEFAULT_OUTPUT = Path("results/phase4b_benchmark.json")
DEFAULT_PLOT = Path("results/phase4b_benchmark.png")


def _parse_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "values must be a non-empty comma-separated unique list"
        )
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "values must be comma-separated numbers"
        ) from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(
            "values must be a non-empty comma-separated unique list"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-family, multi-seed dynamic scheduling benchmark."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sizes", type=_parse_strings, default=("small",))
    parser.add_argument("--loads", type=_parse_floats, default=(0.6, 1.0))
    parser.add_argument("--cvs", type=_parse_floats, default=(0.0, 0.3))
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument("--workflows", type=int, default=6)
    parser.add_argument("--aging-weight", type=float, default=1.0)
    parser.add_argument(
        "--policies",
        type=_parse_policies,
        default=POLICY_NAMES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"also save a benchmark plot (suggested: {DEFAULT_PLOT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_trace_model_config(args.config)
        corpus = load_benchmark_corpus(args.manifest, config)
        benchmark = run_dynamic_benchmark(
            corpus=corpus,
            sizes=args.sizes,
            loads=args.loads,
            cvs=args.cvs,
            replicate_count=args.seed_count,
            workflow_count=args.workflows,
            base_seed=args.base_seed,
            policies=args.policies,
            aging_weight=args.aging_weight,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    payload = {
        "interpretation": {
            "study_type": (
                "multi-family trace-driven dynamic scheduling benchmark"
            ),
            "offered_load": (
                "mean estimated work divided by worker capacity and mean "
                "inter-arrival time; realized utilization can differ"
            ),
            "fairness": (
                "all policies in one sweep cell share arrivals and actual "
                "duration realizations"
            ),
            "timing_note": (
                "scheduler wall seconds are machine-dependent; all other core "
                "simulated metrics are seeded"
            ),
        },
        "corpus": corpus.to_dict(),
        "worker_model": config.to_dict(),
        "benchmark": benchmark.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Benchmark: {corpus.name} | sizes={list(benchmark.sizes)} | "
        f"loads={list(benchmark.loads)} | CVs={list(benchmark.cvs)} | "
        f"replicates={benchmark.replicate_count} | "
        f"policies={len(benchmark.policies)}"
    )
    for aggregate in benchmark.aggregates:
        mean_jct = aggregate.statistics["mean_jct"]
        utilization = aggregate.statistics["average_utilization"]
        print(
            f"  {aggregate.size} load={aggregate.offered_load:.2f} "
            f"CV={aggregate.runtime_cv:.2f} {aggregate.policy}: "
            f"JCT {mean_jct.mean:.3f} +/- "
            f"{mean_jct.ci95_half_width:.3f} s | "
            f"utilization {utilization.mean:.1%}"
        )
    print(f"Valid: {benchmark.is_valid}")
    print(f"JSON: {args.output}")

    if args.plot is not None:
        try:
            output = save_benchmark_plot(benchmark, args.plot)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Benchmark plot: {output}")

    return 0 if benchmark.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
