"""Command-line interface for trace-driven HEFT scheduling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .scheduler import HeftResult, schedule_heft
from .trace_metrics import TraceScheduleMetrics, calculate_trace_metrics
from .trace_model import (
    ModeledTraceWorkflow,
    build_modeled_workflow,
    load_trace_model_config,
)
from .trace_visualization import save_trace_gantt_chart
from .wfcommons import load_wfcommons_trace


DEFAULT_INPUT = Path(
    "data/raw/wfcommons/montage-chameleon-2mass-005d-001.json"
)
DEFAULT_CONFIG = Path("configs/trace_worker_model.json")
DEFAULT_OUTPUT = Path("results/montage_trace_schedule.json")
DEFAULT_PLOT = Path("results/montage_trace_gantt.png")
UPSTREAM_URL = (
    "https://github.com/wfcommons/WfInstances/blob/main/pegasus/montage/"
    "montage-chameleon-2mass-005d-001.json"
)
UPSTREAM_GIT_BLOB = "a9e22f2add751962fc83e83ea1371667977a9975"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HEFT on a WfCommons WfFormat execution trace."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"WfFormat JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"worker model JSON path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"result JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"also save a Gantt chart (suggested: {DEFAULT_PLOT})",
    )
    return parser


def build_trace_payload(
    modeled: ModeledTraceWorkflow,
    result: HeftResult,
    metrics: TraceScheduleMetrics,
    input_sha256: str,
) -> dict[str, object]:
    """Build the deterministic, source-aware trace result document."""

    schedule = []
    for task in sorted(result.schedule):
        entry = result.schedule[task]
        metadata = modeled.task_metadata[task]
        schedule.append(
            {
                **entry.to_dict(),
                "source_id": metadata.source_id,
                "source_name": metadata.source_name,
                "program": metadata.program,
                "observed_runtime_in_seconds": (
                    metadata.observed_runtime_in_seconds
                ),
                "avg_cpu": metadata.avg_cpu,
                "cpu_weight": metadata.cpu_weight,
            }
        )

    priority = [
        {
            "internal_id": task,
            "source_id": modeled.internal_to_source[task],
            "program": modeled.task_metadata[task].program,
            "upward_rank": result.upward_ranks[task],
        }
        for task in result.priority_order
    ]
    edge_report = [
        {
            "parent_internal_id": parent,
            "child_internal_id": child,
            "parent_source_id": modeled.internal_to_source[parent],
            "child_source_id": modeled.internal_to_source[child],
            "shared_file_bytes": modeled.edge_data_bytes[(parent, child)],
            "modeled_communication_seconds": (
                modeled.workflow.communication_cost(parent, child)
            ),
        }
        for parent, child in sorted(modeled.edge_data_bytes)
    ]

    return {
        "source": {
            "dataset": "WfCommons WfInstances",
            "workflow": modeled.trace.name,
            "local_filename": modeled.trace.source_path.name,
            "upstream_url": UPSTREAM_URL,
            "upstream_git_blob": UPSTREAM_GIT_BLOB,
            "sha256": input_sha256,
            "schema_version": modeled.trace.schema_version,
            "runtime_system": modeled.trace.runtime_system,
            "license": "LGPL-3.0",
        },
        "interpretation": {
            "study_type": "trace-driven heterogeneous scheduling simulation",
            "observed_trace_data": (
                "DAG, task runtime, avgCPU, files, file sizes, and source "
                "machine metadata"
            ),
            "simulated_assumptions": (
                "worker compute/I-O speeds and inter-worker bandwidth"
            ),
            "observed_makespan_is_provenance_only": True,
        },
        "trace_summary": modeled.trace.summary(),
        "model_config": modeled.config.to_dict(),
        "task_mapping": [
            modeled.task_report(task) for task in modeled.workflow.tasks
        ],
        "edges": edge_report,
        "algorithm": "HEFT",
        "priority_order": priority,
        "schedule": schedule,
        "modeled_makespan_in_seconds": result.makespan,
        "metrics": metrics.to_dict(),
        "is_valid": result.is_valid,
        "validation_errors": list(result.validation_errors),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        trace = load_wfcommons_trace(args.input)
        config = load_trace_model_config(args.config)
        modeled = build_modeled_workflow(trace, config)
        result = schedule_heft(modeled.workflow)
        metrics = calculate_trace_metrics(modeled, result)
        payload = build_trace_payload(
            modeled,
            result,
            metrics,
            _sha256(args.input),
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assignments = Counter(
        entry.processor for entry in result.schedule.values()
    )
    print(f"Trace: {trace.name} (WfFormat {trace.schema_version})")
    print(
        f"Tasks: {len(trace.tasks)} | Edges: {trace.edge_count} | "
        f"Files: {len(trace.files)}"
    )
    print(
        "Observed Pegasus makespan (provenance only): "
        f"{trace.observed_makespan_in_seconds:.3f} s"
    )
    print(f"Worker model: {config.model_name}")
    print("Assignments:")
    for processor in modeled.workflow.processors:
        print(
            f"  {processor}: {assignments[processor]} tasks, "
            f"{metrics.processor_utilization[processor]:.1%} utilization"
        )
    print(f"Modeled HEFT makespan: {result.makespan:.3f} s")
    print(f"Simulated speedup vs best serial worker: {metrics.simulated_speedup:.3f}x")
    print(
        "Cross-worker transfer: "
        f"{metrics.cross_worker_data_bytes / 1_000_000:.3f} MB "
        f"across {metrics.cross_worker_edge_count} edges"
    )
    print(f"Valid: {result.is_valid}")
    print(f"JSON: {args.output}")

    if args.plot is not None:
        try:
            output = save_trace_gantt_chart(
                modeled,
                result.schedule,
                args.plot,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return 2
        print(f"Gantt chart: {output}")

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
