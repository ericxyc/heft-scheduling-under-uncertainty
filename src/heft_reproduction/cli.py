"""Command-line entry point for reproducing the HEFT paper example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .paper_example import PAPER_DOI, load_paper_example
from .scheduler import schedule_heft
from .visualization import save_gantt_chart


DEFAULT_JSON = Path("results/paper_example_schedule.json")
DEFAULT_PLOT = Path("results/paper_example_gantt.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the HEFT paper's 10-task scheduling example."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JSON,
        help=f"JSON output path (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"also save a Gantt chart (suggested: {DEFAULT_PLOT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workflow = load_paper_example()
    result = schedule_heft(workflow)

    payload = {
        "source": {
            "paper": (
                "Performance-Effective and Low-Complexity Task Scheduling "
                "for Heterogeneous Computing"
            ),
            "doi": PAPER_DOI,
            "fixture": "Figure 3 and Table 1",
        },
        **result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Upward ranks:")
    for task in result.priority_order:
        print(f"  T{task}: {result.upward_ranks[task]:.3f}")
    print(f"Priority order: {', '.join(f'T{task}' for task in result.priority_order)}")
    print("Schedule:")
    for task in result.priority_order:
        entry = result.schedule[task]
        print(
            f"  T{task}: {entry.processor} "
            f"[{entry.start:.3f}, {entry.finish:.3f}]"
        )
    print(f"Makespan: {result.makespan:.3f}")
    print(f"Valid: {result.is_valid}")
    print(f"JSON: {args.output}")

    if args.plot is not None:
        output = save_gantt_chart(workflow, result.schedule, args.plot)
        print(f"Gantt chart: {output}")

    if result.validation_errors:
        for error in result.validation_errors:
            print(f"Validation error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
