"""HEFT ranking, insertion scheduling, and independent validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Mapping

from .models import (
    ProcessorId,
    Schedule,
    ScheduleEntry,
    TaskId,
    Workflow,
    processor_timelines,
)


TOLERANCE = 1e-9


@dataclass(frozen=True)
class HeftResult:
    """Canonical output of one deterministic HEFT run."""

    upward_ranks: Mapping[TaskId, float]
    priority_order: tuple[TaskId, ...]
    schedule: Schedule
    makespan: float
    validation_errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.validation_errors

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": "HEFT",
            "upward_ranks": {
                str(task): self.upward_ranks[task]
                for task in sorted(self.upward_ranks)
            },
            "priority_order": list(self.priority_order),
            "schedule": [
                self.schedule[task].to_dict() for task in sorted(self.schedule)
            ],
            "makespan": self.makespan,
            "is_valid": self.is_valid,
            "validation_errors": list(self.validation_errors),
        }


def compute_upward_ranks(workflow: Workflow) -> dict[TaskId, float]:
    """Calculate HEFT upward ranks using average processor computation costs."""

    ranks: dict[TaskId, float] = {}

    def rank(task: TaskId) -> float:
        if task in ranks:
            return ranks[task]

        successors = workflow.successors(task)
        longest_successor_path = max(
            (
                workflow.communication_cost(task, child) + rank(child)
                for child in successors
            ),
            default=0.0,
        )
        ranks[task] = (
            workflow.mean_computation_cost(task) + longest_successor_path
        )
        return ranks[task]

    for task in reversed(workflow.topological_order()):
        rank(task)
    return {task: ranks[task] for task in workflow.tasks}


def prioritize_tasks(
    workflow: Workflow,
    upward_ranks: Mapping[TaskId, float] | None = None,
) -> tuple[TaskId, ...]:
    """Return tasks in descending upward-rank order with stable ID tie-breaking."""

    ranks = upward_ranks or compute_upward_ranks(workflow)
    if set(ranks) != set(workflow.tasks):
        raise ValueError("upward ranks must cover every task exactly")
    return tuple(
        sorted(
            workflow.tasks,
            key=lambda task: (-round(ranks[task], 12), task),
        )
    )


def dependency_ready_time(
    workflow: Workflow,
    task: TaskId,
    processor: ProcessorId,
    schedule: Schedule,
) -> float:
    """Return the earliest time all inputs can be available on a processor."""

    ready_time = 0.0
    for parent in workflow.predecessors(task):
        if parent not in schedule:
            raise ValueError(
                f"cannot place task {task} before parent {parent} is scheduled"
            )
        parent_entry = schedule[parent]
        transfer = (
            0.0
            if parent_entry.processor == processor
            else workflow.communication_cost(parent, task)
        )
        ready_time = max(ready_time, parent_entry.finish + transfer)
    return ready_time


def earliest_start_on_processor(
    workflow: Workflow,
    task: TaskId,
    processor: ProcessorId,
    schedule: Schedule,
) -> float:
    """Find the first feasible start, including internal processor idle gaps."""

    duration = workflow.computation_cost(task, processor)
    candidate = dependency_ready_time(workflow, task, processor, schedule)
    timeline = processor_timelines(schedule, workflow.processors)[processor]

    for entry in timeline:
        if candidate + duration <= entry.start + TOLERANCE:
            return candidate
        candidate = max(candidate, entry.finish)
    return candidate


def calculate_makespan(schedule: Schedule) -> float:
    """Return the latest task finish time, or zero for an empty schedule."""

    return max((entry.finish for entry in schedule.values()), default=0.0)


def validate_schedule(
    workflow: Workflow,
    schedule: Schedule,
) -> tuple[str, ...]:
    """Independently check coverage, duration, overlap, and dependencies."""

    errors: list[str] = []
    expected_tasks = set(workflow.tasks)
    scheduled_tasks = set(schedule)

    missing = sorted(expected_tasks - scheduled_tasks)
    unexpected = sorted(scheduled_tasks - expected_tasks)
    if missing:
        errors.append(f"missing tasks: {missing}")
    if unexpected:
        errors.append(f"unexpected tasks: {unexpected}")

    for task in sorted(expected_tasks & scheduled_tasks):
        entry = schedule[task]
        if entry.task != task:
            errors.append(
                f"task-key mismatch: key {task} contains task {entry.task}"
            )
        if entry.processor not in workflow.processors:
            errors.append(
                f"task {task} uses unknown processor {entry.processor}"
            )
            continue
        if entry.start < -TOLERANCE or entry.finish < entry.start - TOLERANCE:
            errors.append(
                f"task {task} has invalid interval "
                f"[{entry.start}, {entry.finish}]"
            )
        expected_duration = workflow.computation_cost(task, entry.processor)
        if not isclose(
            entry.duration,
            expected_duration,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            errors.append(
                f"task {task} duration {entry.duration} does not match "
                f"{expected_duration} on {entry.processor}"
            )

    valid_entries = {
        task: entry
        for task, entry in schedule.items()
        if task in expected_tasks and entry.processor in workflow.processors
    }
    timelines = processor_timelines(valid_entries, workflow.processors)
    for processor, entries in timelines.items():
        for previous, current in zip(entries, entries[1:]):
            if current.start < previous.finish - TOLERANCE:
                errors.append(
                    f"processor-overlap on {processor}: tasks "
                    f"{previous.task} and {current.task}"
                )

    for (parent, child), communication in sorted(
        workflow.communication_costs.items()
    ):
        if parent not in valid_entries or child not in valid_entries:
            continue
        parent_entry = valid_entries[parent]
        child_entry = valid_entries[child]
        transfer = (
            0.0
            if parent_entry.processor == child_entry.processor
            else communication
        )
        required_start = parent_entry.finish + transfer
        if child_entry.start < required_start - TOLERANCE:
            errors.append(
                f"dependency violation {parent}->{child}: child starts at "
                f"{child_entry.start}, requires at least {required_start}"
            )

    return tuple(errors)


def schedule_heft(workflow: Workflow) -> HeftResult:
    """Run deterministic insertion-based HEFT for a static workflow DAG."""

    ranks = compute_upward_ranks(workflow)
    priority_order = prioritize_tasks(workflow, ranks)
    schedule: dict[TaskId, ScheduleEntry] = {}
    processor_order = {
        processor: index
        for index, processor in enumerate(workflow.processors)
    }

    for task in priority_order:
        candidates: list[tuple[float, int, float, ProcessorId]] = []
        for processor in workflow.processors:
            start = earliest_start_on_processor(
                workflow, task, processor, schedule
            )
            finish = start + workflow.computation_cost(task, processor)
            candidates.append(
                (finish, processor_order[processor], start, processor)
            )

        finish, _, start, processor = min(candidates)
        schedule[task] = ScheduleEntry(
            task=task,
            processor=processor,
            start=start,
            finish=finish,
        )

    validation_errors = validate_schedule(workflow, schedule)
    return HeftResult(
        upward_ranks=ranks,
        priority_order=priority_order,
        schedule=schedule,
        makespan=calculate_makespan(schedule),
        validation_errors=validation_errors,
    )
