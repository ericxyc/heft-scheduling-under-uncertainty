"""Convert a WfCommons execution trace into a simulated HEFT workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .models import Edge, TaskId, Workflow
from .wfcommons import WfTrace


@dataclass(frozen=True)
class WorkerProfile:
    id: str
    compute_speed: float
    io_speed: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "id": self.id,
            "compute_speed": self.compute_speed,
            "io_speed": self.io_speed,
        }


@dataclass(frozen=True)
class TraceModelConfig:
    model_name: str
    cpu_weight_source: str
    default_cpu_weight: float
    network_bandwidth_bytes_per_second: float
    workers: tuple[WorkerProfile, ...]
    source_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "cpu_weight_source": self.cpu_weight_source,
            "default_cpu_weight": self.default_cpu_weight,
            "network_bandwidth_bytes_per_second": (
                self.network_bandwidth_bytes_per_second
            ),
            "runtime_formula": (
                "observed_runtime * (cpu_weight / compute_speed + "
                "(1 - cpu_weight) / io_speed)"
            ),
            "communication_formula": (
                "shared_file_bytes / network_bandwidth_bytes_per_second"
            ),
            "workers": [worker.to_dict() for worker in self.workers],
        }


@dataclass(frozen=True)
class ModeledTaskMetadata:
    internal_id: TaskId
    source_id: str
    source_name: str
    program: str
    observed_runtime_in_seconds: float
    avg_cpu: float | None
    cpu_weight: float

    @property
    def short_label(self) -> str:
        return f"{self.program}:{self.internal_id}"

    def to_dict(self) -> dict[str, object]:
        return {
            "internal_id": self.internal_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "program": self.program,
            "observed_runtime_in_seconds": self.observed_runtime_in_seconds,
            "avg_cpu": self.avg_cpu,
            "cpu_weight": self.cpu_weight,
        }


@dataclass(frozen=True)
class ModeledTraceWorkflow:
    trace: WfTrace
    config: TraceModelConfig
    workflow: Workflow
    source_to_internal: Mapping[str, TaskId]
    internal_to_source: Mapping[TaskId, str]
    task_metadata: Mapping[TaskId, ModeledTaskMetadata]
    edge_data_bytes: Mapping[Edge, int]

    def task_report(self, task: TaskId) -> dict[str, object]:
        metadata = self.task_metadata[task]
        return {
            **metadata.to_dict(),
            "modeled_computation_costs": {
                processor: self.workflow.computation_cost(task, processor)
                for processor in self.workflow.processors
            },
        }


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{path} must be positive and finite")
    return result


def load_trace_model_config(path: str | Path) -> TraceModelConfig:
    """Load and validate the deterministic trace simulation configuration."""

    source_path = Path(path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read model config {source_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("model config root must be an object")

    model_name = raw.get("model_name")
    cpu_weight_source = raw.get("cpu_weight_source")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(cpu_weight_source, str) or not cpu_weight_source:
        raise ValueError("cpu_weight_source must be a non-empty string")

    default_weight = raw.get("default_cpu_weight")
    if isinstance(default_weight, bool) or not isinstance(
        default_weight, (int, float)
    ):
        raise ValueError("default_cpu_weight must be a number")
    default_weight = float(default_weight)
    if not 0 <= default_weight <= 1:
        raise ValueError("default_cpu_weight must be between zero and one")

    workers_raw = raw.get("workers")
    if not isinstance(workers_raw, list) or not workers_raw:
        raise ValueError("workers must be a non-empty array")
    workers: list[WorkerProfile] = []
    for index, value in enumerate(workers_raw):
        if not isinstance(value, dict):
            raise ValueError(f"workers[{index}] must be an object")
        worker_id = value.get("id")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError(f"workers[{index}].id must be a non-empty string")
        workers.append(
            WorkerProfile(
                id=worker_id,
                compute_speed=_positive_number(
                    value.get("compute_speed"),
                    f"workers[{index}].compute_speed",
                ),
                io_speed=_positive_number(
                    value.get("io_speed"),
                    f"workers[{index}].io_speed",
                ),
            )
        )
    if len({worker.id for worker in workers}) != len(workers):
        raise ValueError("worker IDs must be unique")

    return TraceModelConfig(
        model_name=model_name,
        cpu_weight_source=cpu_weight_source,
        default_cpu_weight=default_weight,
        network_bandwidth_bytes_per_second=_positive_number(
            raw.get("network_bandwidth_bytes_per_second"),
            "network_bandwidth_bytes_per_second",
        ),
        workers=tuple(workers),
        source_path=source_path,
    )


def _cpu_weight(avg_cpu: float | None, default: float) -> float:
    if avg_cpu is None:
        return default
    return min(1.0, max(0.0, avg_cpu / 100.0))


def modeled_runtime(
    observed_runtime: float,
    cpu_weight: float,
    worker: WorkerProfile,
) -> float:
    """Apply the documented CPU/I/O proxy formula."""

    return observed_runtime * (
        cpu_weight / worker.compute_speed
        + (1.0 - cpu_weight) / worker.io_speed
    )


def build_modeled_workflow(
    trace: WfTrace,
    config: TraceModelConfig,
) -> ModeledTraceWorkflow:
    """Create a deterministic heterogeneous HEFT workflow from a trace."""

    source_ids = sorted(trace.tasks)
    source_to_internal = {
        source_id: index
        for index, source_id in enumerate(source_ids, start=1)
    }
    internal_to_source = {
        internal: source for source, internal in source_to_internal.items()
    }
    processors = tuple(worker.id for worker in config.workers)

    computation_costs: dict[TaskId, dict[str, float]] = {}
    metadata: dict[TaskId, ModeledTaskMetadata] = {}
    for source_id in source_ids:
        internal_id = source_to_internal[source_id]
        specification = trace.tasks[source_id]
        execution = trace.executions[source_id]
        weight = _cpu_weight(execution.avg_cpu, config.default_cpu_weight)
        computation_costs[internal_id] = {
            worker.id: modeled_runtime(
                execution.runtime_in_seconds,
                weight,
                worker,
            )
            for worker in config.workers
        }
        metadata[internal_id] = ModeledTaskMetadata(
            internal_id=internal_id,
            source_id=source_id,
            source_name=specification.name,
            program=execution.program,
            observed_runtime_in_seconds=execution.runtime_in_seconds,
            avg_cpu=execution.avg_cpu,
            cpu_weight=weight,
        )

    communication_costs: dict[Edge, float] = {}
    edge_data_bytes: dict[Edge, int] = {}
    for parent_source in source_ids:
        parent_internal = source_to_internal[parent_source]
        for child_source in sorted(trace.tasks[parent_source].children):
            child_internal = source_to_internal[child_source]
            edge = (parent_internal, child_internal)
            shared_bytes = trace.shared_bytes(parent_source, child_source)
            edge_data_bytes[edge] = shared_bytes
            communication_costs[edge] = (
                shared_bytes / config.network_bandwidth_bytes_per_second
            )

    workflow = Workflow(
        tasks=tuple(sorted(internal_to_source)),
        processors=processors,
        computation_costs=computation_costs,
        communication_costs=communication_costs,
    )
    return ModeledTraceWorkflow(
        trace=trace,
        config=config,
        workflow=workflow,
        source_to_internal=source_to_internal,
        internal_to_source=internal_to_source,
        task_metadata=metadata,
        edge_data_bytes=edge_data_bytes,
    )
