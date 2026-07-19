"""Data models for dynamic multi-workflow scheduling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .models import Edge, ProcessorId, TaskId, Workflow


TaskRef = tuple[str, TaskId]

ONLINE_GREEDY = "online-greedy-eft"
STATIC_HEFT = "per-workflow-static-heft"
ROLLING_HEFT = "rolling-heft"
AGING_ROLLING_HEFT = "aging-rolling-heft"
SHORTEST_REMAINING_WORK = "shortest-remaining-work"
POLICY_NAMES = (
    ONLINE_GREEDY,
    STATIC_HEFT,
    ROLLING_HEFT,
    AGING_ROLLING_HEFT,
    SHORTEST_REMAINING_WORK,
)


@dataclass(frozen=True)
class WorkflowTemplate:
    """One trace-derived workflow shape available to the scenario builder."""

    name: str
    workflow: Workflow
    source_task_ids: Mapping[TaskId, str]
    programs: Mapping[TaskId, str]
    edge_data_bytes: Mapping[Edge, int]
    source_filename: str

    def __post_init__(self) -> None:
        task_set = set(self.workflow.tasks)
        if not self.name:
            raise ValueError("workflow template name must be non-empty")
        if set(self.source_task_ids) != task_set:
            raise ValueError("source task IDs must cover every template task")
        if set(self.programs) != task_set:
            raise ValueError("programs must cover every template task")
        if set(self.edge_data_bytes) != set(
            self.workflow.communication_costs
        ):
            raise ValueError("edge bytes must cover every workflow edge")


@dataclass(frozen=True)
class DynamicWorkflowInstance:
    """One complete workflow DAG arriving during a dynamic scenario."""

    id: str
    template: WorkflowTemplate
    arrival_time: float
    actual_workflow: Workflow
    runtime_seed: int

    def __post_init__(self) -> None:
        estimated = self.template.workflow
        if not self.id:
            raise ValueError("workflow instance ID must be non-empty")
        if not isfinite(self.arrival_time) or self.arrival_time < 0:
            raise ValueError(
                "workflow arrival time must be finite and non-negative"
            )
        if estimated.tasks != self.actual_workflow.tasks:
            raise ValueError("actual workflow tasks must match the template")
        if estimated.processors != self.actual_workflow.processors:
            raise ValueError("actual workflow workers must match the template")
        if (
            estimated.communication_costs
            != self.actual_workflow.communication_costs
        ):
            raise ValueError("actual workflow dependencies must match the template")

    def ref(self, task: TaskId) -> TaskRef:
        return (self.id, task)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "template": self.template.name,
            "arrival_time": self.arrival_time,
            "runtime_seed": self.runtime_seed,
            "task_count": len(self.template.workflow.tasks),
            "edge_count": len(self.template.workflow.communication_costs),
        }


@dataclass(frozen=True)
class DynamicScenario:
    """Immutable arrivals and realized durations shared by all policies."""

    instances: tuple[DynamicWorkflowInstance, ...]
    runtime_cv: float
    mean_interarrival_time: float
    master_seed: int

    def __post_init__(self) -> None:
        if not self.instances:
            raise ValueError("dynamic scenario must contain at least one workflow")
        if not isfinite(self.runtime_cv) or self.runtime_cv < 0:
            raise ValueError("runtime CV must be finite and non-negative")
        if (
            not isfinite(self.mean_interarrival_time)
            or self.mean_interarrival_time < 0
        ):
            raise ValueError(
                "mean inter-arrival time must be finite and non-negative"
            )
        if len({instance.id for instance in self.instances}) != len(
            self.instances
        ):
            raise ValueError("workflow instance IDs must be unique")
        processors = self.instances[0].template.workflow.processors
        if any(
            instance.template.workflow.processors != processors
            for instance in self.instances
        ):
            raise ValueError("all workflow templates must use the same workers")
        arrivals = [instance.arrival_time for instance in self.instances]
        if arrivals != sorted(arrivals):
            raise ValueError("workflow instances must be sorted by arrival time")

    @property
    def processors(self) -> tuple[ProcessorId, ...]:
        return self.instances[0].template.workflow.processors

    @property
    def task_count(self) -> int:
        return sum(
            len(instance.template.workflow.tasks)
            for instance in self.instances
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_count": len(self.instances),
            "task_count": self.task_count,
            "runtime_cv": self.runtime_cv,
            "mean_interarrival_time": self.mean_interarrival_time,
            "master_seed": self.master_seed,
            "workers": list(self.processors),
            "instances": [instance.to_dict() for instance in self.instances],
        }


@dataclass(frozen=True)
class DynamicTaskExecution:
    """One committed non-preemptive task execution."""

    workflow_id: str
    template_name: str
    task: TaskId
    source_id: str
    program: str
    processor: ProcessorId
    start: float
    finish: float
    estimated_duration: float
    actual_duration: float
    input_ready_time: float

    @property
    def ref(self) -> TaskRef:
        return (self.workflow_id, self.task)

    @property
    def queue_wait(self) -> float:
        return self.start - self.input_ready_time

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "template": self.template_name,
            "task": self.task,
            "source_id": self.source_id,
            "program": self.program,
            "processor": self.processor,
            "start": self.start,
            "finish": self.finish,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "input_ready_time": self.input_ready_time,
            "queue_wait": self.queue_wait,
        }


@dataclass(frozen=True)
class DynamicWorkflowExecution:
    """Completion metrics for one workflow instance."""

    workflow_id: str
    template_name: str
    arrival_time: float
    first_start_time: float
    completion_time: float
    task_count: int

    @property
    def response_time(self) -> float:
        return self.first_start_time - self.arrival_time

    @property
    def jct(self) -> float:
        return self.completion_time - self.arrival_time

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "template": self.template_name,
            "arrival_time": self.arrival_time,
            "first_start_time": self.first_start_time,
            "completion_time": self.completion_time,
            "response_time": self.response_time,
            "jct": self.jct,
            "task_count": self.task_count,
        }


@dataclass(frozen=True)
class DynamicMetrics:
    """Aggregate metrics for one policy on one immutable scenario."""

    workflow_count: int
    task_count: int
    simulation_horizon: float
    mean_jct: float
    p95_jct: float
    mean_response_time: float
    mean_task_queue_wait: float
    p95_task_queue_wait: float
    throughput_workflows_per_second: float
    processor_busy_time: Mapping[ProcessorId, float]
    processor_utilization: Mapping[ProcessorId, float]
    average_utilization: float
    cross_worker_edge_count: int
    cross_worker_data_bytes: int
    cross_worker_communication_seconds: float
    scheduling_rounds: int
    committed_decisions: int
    candidate_evaluations: int
    scheduler_wall_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_count": self.workflow_count,
            "task_count": self.task_count,
            "simulation_horizon": self.simulation_horizon,
            "mean_jct": self.mean_jct,
            "p95_jct": self.p95_jct,
            "mean_response_time": self.mean_response_time,
            "mean_task_queue_wait": self.mean_task_queue_wait,
            "p95_task_queue_wait": self.p95_task_queue_wait,
            "throughput_workflows_per_second": (
                self.throughput_workflows_per_second
            ),
            "processor_busy_time": dict(self.processor_busy_time),
            "processor_utilization": dict(self.processor_utilization),
            "average_utilization": self.average_utilization,
            "cross_worker_edge_count": self.cross_worker_edge_count,
            "cross_worker_data_bytes": self.cross_worker_data_bytes,
            "cross_worker_communication_seconds": (
                self.cross_worker_communication_seconds
            ),
            "scheduling_rounds": self.scheduling_rounds,
            "committed_decisions": self.committed_decisions,
            "candidate_evaluations": self.candidate_evaluations,
            "scheduler_wall_seconds": self.scheduler_wall_seconds,
        }


@dataclass(frozen=True)
class DynamicSimulationResult:
    """A validated policy execution over a dynamic scenario."""

    policy: str
    tasks: tuple[DynamicTaskExecution, ...]
    workflows: tuple[DynamicWorkflowExecution, ...]
    metrics: DynamicMetrics
    event_count: int
    validation_errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.validation_errors

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "event_count": self.event_count,
            "metrics": self.metrics.to_dict(),
            "workflows": [workflow.to_dict() for workflow in self.workflows],
            "tasks": [task.to_dict() for task in self.tasks],
            "is_valid": self.is_valid,
            "validation_errors": list(self.validation_errors),
        }
