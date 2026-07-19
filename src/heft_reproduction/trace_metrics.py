"""Metrics for a trace-driven HEFT schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .scheduler import HeftResult
from .trace_model import ModeledTraceWorkflow


@dataclass(frozen=True)
class TraceScheduleMetrics:
    processor_busy_time: Mapping[str, float]
    processor_utilization: Mapping[str, float]
    average_utilization: float
    serial_runtime_by_processor: Mapping[str, float]
    best_serial_runtime: float
    simulated_speedup: float
    cross_worker_edge_count: int
    cross_worker_data_bytes: int
    cross_worker_communication_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "processor_busy_time": dict(self.processor_busy_time),
            "processor_utilization": dict(self.processor_utilization),
            "average_utilization": self.average_utilization,
            "serial_runtime_by_processor": dict(
                self.serial_runtime_by_processor
            ),
            "best_serial_runtime": self.best_serial_runtime,
            "simulated_speedup": self.simulated_speedup,
            "cross_worker_edge_count": self.cross_worker_edge_count,
            "cross_worker_data_bytes": self.cross_worker_data_bytes,
            "cross_worker_communication_seconds": (
                self.cross_worker_communication_seconds
            ),
        }


def calculate_trace_metrics(
    modeled: ModeledTraceWorkflow,
    result: HeftResult,
) -> TraceScheduleMetrics:
    """Calculate deterministic utilization and communication statistics."""

    workflow = modeled.workflow
    busy_time = {processor: 0.0 for processor in workflow.processors}
    for entry in result.schedule.values():
        busy_time[entry.processor] += entry.duration

    if result.makespan <= 0:
        utilization = {processor: 0.0 for processor in workflow.processors}
        average_utilization = 0.0
    else:
        utilization = {
            processor: busy_time[processor] / result.makespan
            for processor in workflow.processors
        }
        average_utilization = sum(busy_time.values()) / (
            result.makespan * len(workflow.processors)
        )

    serial_runtime = {
        processor: sum(
            workflow.computation_cost(task, processor)
            for task in workflow.tasks
        )
        for processor in workflow.processors
    }
    best_serial = min(serial_runtime.values())
    speedup = best_serial / result.makespan if result.makespan > 0 else 0.0

    cross_edges = 0
    cross_bytes = 0
    cross_seconds = 0.0
    for edge, data_bytes in modeled.edge_data_bytes.items():
        parent, child = edge
        if (
            result.schedule[parent].processor
            != result.schedule[child].processor
        ):
            cross_edges += 1
            cross_bytes += data_bytes
            cross_seconds += workflow.communication_cost(parent, child)

    return TraceScheduleMetrics(
        processor_busy_time=busy_time,
        processor_utilization=utilization,
        average_utilization=average_utilization,
        serial_runtime_by_processor=serial_runtime,
        best_serial_runtime=best_serial,
        simulated_speedup=speedup,
        cross_worker_edge_count=cross_edges,
        cross_worker_data_bytes=cross_bytes,
        cross_worker_communication_seconds=cross_seconds,
    )
