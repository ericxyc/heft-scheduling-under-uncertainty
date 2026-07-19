"""Core workflow and schedule data models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


TaskId = int
ProcessorId = str
Edge = tuple[TaskId, TaskId]


@dataclass(frozen=True)
class Workflow:
    """A static DAG with processor-specific computation costs."""

    tasks: tuple[TaskId, ...]
    processors: tuple[ProcessorId, ...]
    computation_costs: Mapping[TaskId, Mapping[ProcessorId, float]]
    communication_costs: Mapping[Edge, float]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.tasks:
            raise ValueError("workflow must contain at least one task")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("task IDs must be unique")
        if not self.processors:
            raise ValueError("workflow must contain at least one processor")
        if len(set(self.processors)) != len(self.processors):
            raise ValueError("processor IDs must be unique")

        task_set = set(self.tasks)
        processor_set = set(self.processors)
        if set(self.computation_costs) != task_set:
            raise ValueError("computation costs must cover every task exactly")

        for task in self.tasks:
            costs = self.computation_costs[task]
            if set(costs) != processor_set:
                raise ValueError(
                    f"task {task} must have one cost for every processor"
                )
            for processor, cost in costs.items():
                if not isfinite(cost) or cost <= 0:
                    raise ValueError(
                        f"task {task} has invalid cost {cost} on {processor}"
                    )

        for (parent, child), cost in self.communication_costs.items():
            if parent not in task_set or child not in task_set:
                raise ValueError(
                    f"edge ({parent}, {child}) references an unknown task"
                )
            if parent == child:
                raise ValueError("self dependencies are not allowed")
            if not isfinite(cost) or cost < 0:
                raise ValueError(
                    f"edge ({parent}, {child}) has invalid cost {cost}"
                )

        self.topological_order()

    def predecessors(self, task: TaskId) -> tuple[TaskId, ...]:
        self._require_task(task)
        return tuple(
            sorted(
                parent
                for parent, child in self.communication_costs
                if child == task
            )
        )

    def successors(self, task: TaskId) -> tuple[TaskId, ...]:
        self._require_task(task)
        return tuple(
            sorted(
                child
                for parent, child in self.communication_costs
                if parent == task
            )
        )

    def communication_cost(self, parent: TaskId, child: TaskId) -> float:
        try:
            return float(self.communication_costs[(parent, child)])
        except KeyError as exc:
            raise KeyError(f"no dependency edge ({parent}, {child})") from exc

    def computation_cost(
        self, task: TaskId, processor: ProcessorId
    ) -> float:
        self._require_task(task)
        if processor not in self.processors:
            raise KeyError(f"unknown processor {processor}")
        return float(self.computation_costs[task][processor])

    def mean_computation_cost(self, task: TaskId) -> float:
        self._require_task(task)
        return sum(self.computation_costs[task].values()) / len(
            self.processors
        )

    def topological_order(self) -> tuple[TaskId, ...]:
        indegree = {task: 0 for task in self.tasks}
        children = {task: [] for task in self.tasks}
        for parent, child in self.communication_costs:
            indegree[child] += 1
            children[parent].append(child)

        ready = sorted(task for task, degree in indegree.items() if degree == 0)
        order: list[TaskId] = []
        while ready:
            task = ready.pop(0)
            order.append(task)
            for child in sorted(children[task]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()

        if len(order) != len(self.tasks):
            raise ValueError("workflow dependencies must form a DAG")
        return tuple(order)

    def _require_task(self, task: TaskId) -> None:
        if task not in self.computation_costs:
            raise KeyError(f"unknown task {task}")


@dataclass(frozen=True)
class ScheduleEntry:
    """A committed placement of one task on one processor."""

    task: TaskId
    processor: ProcessorId
    start: float
    finish: float

    @property
    def duration(self) -> float:
        return self.finish - self.start

    def to_dict(self) -> dict[str, int | str | float]:
        return {
            "task": self.task,
            "processor": self.processor,
            "start": self.start,
            "finish": self.finish,
            "duration": self.duration,
        }


Schedule = Mapping[TaskId, ScheduleEntry]


def processor_timelines(
    schedule: Schedule, processors: tuple[ProcessorId, ...]
) -> dict[ProcessorId, list[ScheduleEntry]]:
    timelines = {processor: [] for processor in processors}
    for entry in schedule.values():
        if entry.processor not in timelines:
            raise ValueError(f"schedule uses unknown processor {entry.processor}")
        timelines[entry.processor].append(entry)
    for entries in timelines.values():
        entries.sort(key=lambda item: (item.start, item.finish, item.task))
    return timelines
