"""Event-driven non-preemptive simulation of dynamic workflow arrivals."""

from __future__ import annotations

from collections import defaultdict
from math import isclose
from statistics import fmean
from time import perf_counter
from typing import Mapping

from .dynamic_models import (
    ONLINE_GREEDY,
    POLICY_NAMES,
    STATIC_HEFT,
    DynamicMetrics,
    DynamicScenario,
    DynamicSimulationResult,
    DynamicTaskExecution,
    DynamicWorkflowExecution,
    DynamicWorkflowInstance,
    TaskRef,
)
from .dynamic_policies import SchedulingCandidate, choose_candidate
from .models import ProcessorId, TaskId
from .scheduler import compute_upward_ranks, schedule_heft


TOLERANCE = 1e-9
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _input_ready_time(
    instance: DynamicWorkflowInstance,
    task: TaskId,
    processor: ProcessorId,
    status: Mapping[TaskRef, str],
    executions: Mapping[TaskRef, DynamicTaskExecution],
) -> float | None:
    """Return worker-specific input availability, or None if a parent runs."""

    workflow = instance.template.workflow
    parents = workflow.predecessors(task)
    if any(status[instance.ref(parent)] != COMPLETED for parent in parents):
        return None
    return max(
        (
            executions[instance.ref(parent)].finish
            + (
                0.0
                if executions[instance.ref(parent)].processor == processor
                else workflow.communication_cost(parent, task)
            )
            for parent in parents
        ),
        default=instance.arrival_time,
    )


def _static_order_allows(
    instance: DynamicWorkflowInstance,
    task: TaskId,
    processor: ProcessorId,
    status: Mapping[TaskRef, str],
    static_orders: Mapping[tuple[str, ProcessorId], tuple[TaskId, ...]],
    static_assignment: Mapping[TaskRef, ProcessorId],
) -> bool:
    if static_assignment[instance.ref(task)] != processor:
        return False
    order = static_orders[(instance.id, processor)]
    for ordered_task in order:
        ordered_status = status[instance.ref(ordered_task)]
        if ordered_status == COMPLETED:
            continue
        return ordered_task == task and ordered_status == PENDING
    return False


def _build_candidates(
    now: float,
    scenario: DynamicScenario,
    arrived_ids: set[str],
    status: Mapping[TaskRef, str],
    executions: Mapping[TaskRef, DynamicTaskExecution],
    running_by_processor: Mapping[ProcessorId, TaskRef],
    ranks: Mapping[str, Mapping[TaskId, float]],
    baseline_makespans: Mapping[str, float],
    aging_weight: float,
    static_orders: Mapping[tuple[str, ProcessorId], tuple[TaskId, ...]],
    static_assignment: Mapping[TaskRef, ProcessorId],
    static_planned_start: Mapping[TaskRef, float],
) -> tuple[SchedulingCandidate, ...]:
    processor_indices = {
        processor: index
        for index, processor in enumerate(scenario.processors)
    }
    idle_processors = tuple(
        processor
        for processor in scenario.processors
        if processor not in running_by_processor
    )
    candidates: list[SchedulingCandidate] = []
    remaining_work = {
        instance.id: sum(
            instance.template.workflow.mean_computation_cost(task)
            for task in instance.template.workflow.tasks
            if status[instance.ref(task)] == PENDING
        )
        for instance in scenario.instances
        if instance.id in arrived_ids
    }
    for instance in scenario.instances:
        if instance.id not in arrived_ids:
            continue
        workflow = instance.template.workflow
        maximum_rank = max(ranks[instance.id].values())
        normalized_age = (
            max(0.0, now - instance.arrival_time)
            / baseline_makespans[instance.id]
        )
        for task in workflow.tasks:
            ref = instance.ref(task)
            if status[ref] != PENDING:
                continue
            for processor in idle_processors:
                input_ready = _input_ready_time(
                    instance,
                    task,
                    processor,
                    status,
                    executions,
                )
                if input_ready is None or input_ready > now + TOLERANCE:
                    continue
                candidates.append(
                    SchedulingCandidate(
                        ref=ref,
                        processor=processor,
                        processor_index=processor_indices[processor],
                        workflow_arrival=instance.arrival_time,
                        estimated_finish=(
                            now + workflow.computation_cost(task, processor)
                        ),
                        upward_rank=ranks[instance.id][task],
                        normalized_upward_rank=(
                            ranks[instance.id][task] / maximum_rank
                        ),
                        aging_score=(
                            ranks[instance.id][task] / maximum_rank
                            + aging_weight * normalized_age
                        ),
                        workflow_remaining_work=remaining_work[instance.id],
                        static_allowed=_static_order_allows(
                            instance,
                            task,
                            processor,
                            status,
                            static_orders,
                            static_assignment,
                        ),
                        static_absolute_planned_start=(
                            instance.arrival_time + static_planned_start[ref]
                        ),
                    )
                )
    return tuple(candidates)


def _next_data_availability(
    now: float,
    scenario: DynamicScenario,
    arrived_ids: set[str],
    status: Mapping[TaskRef, str],
    executions: Mapping[TaskRef, DynamicTaskExecution],
    running_by_processor: Mapping[ProcessorId, TaskRef],
    policy: str,
    static_assignment: Mapping[TaskRef, ProcessorId],
    static_orders: Mapping[tuple[str, ProcessorId], tuple[TaskId, ...]],
) -> float | None:
    """Find the next transfer completion relevant to an idle worker."""

    idle_processors = tuple(
        processor
        for processor in scenario.processors
        if processor not in running_by_processor
    )
    future: list[float] = []
    for instance in scenario.instances:
        if instance.id not in arrived_ids:
            continue
        for task in instance.template.workflow.tasks:
            ref = instance.ref(task)
            if status[ref] != PENDING:
                continue
            for processor in idle_processors:
                if policy == STATIC_HEFT and not _static_order_allows(
                    instance,
                    task,
                    processor,
                    status,
                    static_orders,
                    static_assignment,
                ):
                    continue
                input_ready = _input_ready_time(
                    instance,
                    task,
                    processor,
                    status,
                    executions,
                )
                if input_ready is not None and input_ready > now + TOLERANCE:
                    future.append(input_ready)
    return min(future) if future else None


def _validate_dynamic_schedule(
    scenario: DynamicScenario,
    executions: Mapping[TaskRef, DynamicTaskExecution],
) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        instance.ref(task)
        for instance in scenario.instances
        for task in instance.template.workflow.tasks
    }
    actual = set(executions)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(f"missing tasks: {missing}")
    if unexpected:
        errors.append(f"unexpected tasks: {unexpected}")

    instance_by_id = {
        instance.id: instance for instance in scenario.instances
    }
    for ref in sorted(expected & actual):
        entry = executions[ref]
        instance = instance_by_id[ref[0]]
        if entry.start < instance.arrival_time - TOLERANCE:
            errors.append(f"task {ref} starts before workflow arrival")
        expected_duration = instance.actual_workflow.computation_cost(
            ref[1],
            entry.processor,
        )
        if not isclose(
            entry.actual_duration,
            expected_duration,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            errors.append(f"task {ref} has the wrong actual duration")
        if entry.finish < entry.start - TOLERANCE:
            errors.append(f"task {ref} has a negative execution interval")

    by_processor: dict[ProcessorId, list[DynamicTaskExecution]] = {
        processor: [] for processor in scenario.processors
    }
    for entry in executions.values():
        by_processor[entry.processor].append(entry)
    for processor, timeline in by_processor.items():
        timeline.sort(
            key=lambda entry: (
                entry.start,
                entry.finish,
                entry.workflow_id,
                entry.task,
            )
        )
        for previous, current in zip(timeline, timeline[1:]):
            if current.start < previous.finish - TOLERANCE:
                errors.append(
                    f"worker overlap on {processor}: "
                    f"{previous.ref} and {current.ref}"
                )

    for instance in scenario.instances:
        workflow = instance.template.workflow
        for (parent, child), communication in workflow.communication_costs.items():
            parent_ref = instance.ref(parent)
            child_ref = instance.ref(child)
            if parent_ref not in executions or child_ref not in executions:
                continue
            parent_entry = executions[parent_ref]
            child_entry = executions[child_ref]
            transfer = (
                0.0
                if parent_entry.processor == child_entry.processor
                else communication
            )
            required_start = parent_entry.finish + transfer
            if child_entry.start < required_start - TOLERANCE:
                errors.append(
                    f"dependency violation {parent_ref}->{child_ref}: "
                    f"{child_entry.start} < {required_start}"
                )
    return tuple(errors)


def _build_metrics(
    scenario: DynamicScenario,
    executions: Mapping[TaskRef, DynamicTaskExecution],
    scheduling_rounds: int,
    committed_decisions: int,
    candidate_evaluations: int,
    scheduler_wall_seconds: float,
) -> tuple[tuple[DynamicWorkflowExecution, ...], DynamicMetrics]:
    workflow_results: list[DynamicWorkflowExecution] = []
    for instance in scenario.instances:
        entries = [
            executions[instance.ref(task)]
            for task in instance.template.workflow.tasks
        ]
        workflow_results.append(
            DynamicWorkflowExecution(
                workflow_id=instance.id,
                template_name=instance.template.name,
                arrival_time=instance.arrival_time,
                first_start_time=min(entry.start for entry in entries),
                completion_time=max(entry.finish for entry in entries),
                task_count=len(entries),
            )
        )

    first_arrival = min(instance.arrival_time for instance in scenario.instances)
    last_completion = max(
        workflow.completion_time for workflow in workflow_results
    )
    horizon = last_completion - first_arrival
    jcts = [workflow.jct for workflow in workflow_results]
    responses = [workflow.response_time for workflow in workflow_results]
    waits = [entry.queue_wait for entry in executions.values()]

    busy_time = {processor: 0.0 for processor in scenario.processors}
    for entry in executions.values():
        busy_time[entry.processor] += entry.actual_duration
    utilization = {
        processor: (
            busy_time[processor] / horizon if horizon > 0 else 0.0
        )
        for processor in scenario.processors
    }
    average_utilization = (
        sum(busy_time.values()) / (horizon * len(scenario.processors))
        if horizon > 0
        else 0.0
    )

    cross_edges = 0
    cross_bytes = 0
    cross_seconds = 0.0
    for instance in scenario.instances:
        for edge, data_bytes in instance.template.edge_data_bytes.items():
            parent, child = edge
            if (
                executions[instance.ref(parent)].processor
                != executions[instance.ref(child)].processor
            ):
                cross_edges += 1
                cross_bytes += data_bytes
                cross_seconds += (
                    instance.template.workflow.communication_cost(*edge)
                )

    metrics = DynamicMetrics(
        workflow_count=len(workflow_results),
        task_count=len(executions),
        simulation_horizon=horizon,
        mean_jct=fmean(jcts),
        p95_jct=_percentile(jcts, 0.95),
        mean_response_time=fmean(responses),
        mean_task_queue_wait=fmean(waits),
        p95_task_queue_wait=_percentile(waits, 0.95),
        throughput_workflows_per_second=(
            len(workflow_results) / horizon if horizon > 0 else 0.0
        ),
        processor_busy_time=busy_time,
        processor_utilization=utilization,
        average_utilization=average_utilization,
        cross_worker_edge_count=cross_edges,
        cross_worker_data_bytes=cross_bytes,
        cross_worker_communication_seconds=cross_seconds,
        scheduling_rounds=scheduling_rounds,
        committed_decisions=committed_decisions,
        candidate_evaluations=candidate_evaluations,
        scheduler_wall_seconds=scheduler_wall_seconds,
    )
    return tuple(workflow_results), metrics


class DynamicSchedulingCore:
    """Pausable event state shared by heuristic and learned policies."""

    def __init__(
        self,
        scenario: DynamicScenario,
        aging_weight: float = 1.0,
    ) -> None:
        if aging_weight < 0:
            raise ValueError("aging weight must be non-negative")

        self.scenario = scenario
        self.aging_weight = aging_weight
        self.status: dict[TaskRef, str] = {
            instance.ref(task): PENDING
            for instance in scenario.instances
            for task in instance.template.workflow.tasks
        }
        self.instances = {
            instance.id: instance for instance in scenario.instances
        }
        self.ranks = {
            instance.id: compute_upward_ranks(instance.template.workflow)
            for instance in scenario.instances
        }
        static_results = {
            instance.id: schedule_heft(instance.template.workflow)
            for instance in scenario.instances
        }
        if any(not result.is_valid for result in static_results.values()):
            raise ValueError("a per-workflow static HEFT plan is invalid")
        self.baseline_makespans = {
            instance_id: result.makespan
            for instance_id, result in static_results.items()
        }

        self.static_assignment: dict[TaskRef, ProcessorId] = {}
        self.static_planned_start: dict[TaskRef, float] = {}
        static_orders_lists: dict[
            tuple[str, ProcessorId], list[tuple[float, TaskId]]
        ] = defaultdict(list)
        for instance in scenario.instances:
            result = static_results[instance.id]
            for task, entry in result.schedule.items():
                ref = instance.ref(task)
                self.static_assignment[ref] = entry.processor
                self.static_planned_start[ref] = entry.start
                static_orders_lists[(instance.id, entry.processor)].append(
                    (entry.start, task)
                )
            for processor in scenario.processors:
                static_orders_lists.setdefault((instance.id, processor), [])
        self.static_orders = {
            key: tuple(task for _, task in sorted(values))
            for key, values in static_orders_lists.items()
        }

        self.arrived_ids: set[str] = set()
        self.running_by_processor: dict[ProcessorId, TaskRef] = {}
        self.executions: dict[TaskRef, DynamicTaskExecution] = {}
        self.now = 0.0
        self.event_count = 0
        self.scheduling_rounds = 0
        self.committed_decisions = 0
        self.candidate_evaluations = 0
        self.scheduler_wall_seconds = 0.0
        self._processed_event_time: float | None = None

    @property
    def is_complete(self) -> bool:
        return all(value == COMPLETED for value in self.status.values())

    @property
    def active_workflow_count(self) -> int:
        return sum(
            instance.id in self.arrived_ids
            and any(
                self.status[instance.ref(task)] != COMPLETED
                for task in instance.template.workflow.tasks
            )
            for instance in self.scenario.instances
        )

    def process_current_events(self) -> None:
        """Apply simultaneous completions and arrivals at the current clock."""

        if (
            self._processed_event_time is not None
            and isclose(
                self._processed_event_time,
                self.now,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
        ):
            return
        self.event_count += 1
        completed_processors = [
            processor
            for processor, ref in self.running_by_processor.items()
            if self.executions[ref].finish <= self.now + TOLERANCE
        ]
        for processor in completed_processors:
            ref = self.running_by_processor.pop(processor)
            self.status[ref] = COMPLETED

        for instance in self.scenario.instances:
            if (
                instance.id not in self.arrived_ids
                and instance.arrival_time <= self.now + TOLERANCE
            ):
                self.arrived_ids.add(instance.id)
        self._processed_event_time = self.now

    def candidates(self) -> tuple[SchedulingCandidate, ...]:
        """Return all task-worker pairs feasible at the current time."""

        self.process_current_events()
        return _build_candidates(
            self.now,
            self.scenario,
            self.arrived_ids,
            self.status,
            self.executions,
            self.running_by_processor,
            self.ranks,
            self.baseline_makespans,
            self.aging_weight,
            self.static_orders,
            self.static_assignment,
            self.static_planned_start,
        )

    def choose(self, policy: str) -> SchedulingCandidate | None:
        """Run one existing heuristic decision and record its search effort."""

        start = perf_counter()
        candidates = self.candidates()
        decision, evaluated = choose_candidate(policy, candidates)
        self.scheduler_wall_seconds += perf_counter() - start
        self.scheduling_rounds += 1
        self.candidate_evaluations += evaluated
        return decision

    def record_external_decision(
        self,
        candidate_evaluations: int,
        wall_seconds: float = 0.0,
    ) -> None:
        """Record work performed by an external policy such as an RL model."""

        if candidate_evaluations < 0 or wall_seconds < 0:
            raise ValueError("decision effort must be non-negative")
        self.scheduling_rounds += 1
        self.candidate_evaluations += candidate_evaluations
        self.scheduler_wall_seconds += wall_seconds

    def commit(self, decision: SchedulingCandidate) -> DynamicTaskExecution:
        """Start one currently feasible task non-preemptively."""

        if decision not in self.candidates():
            raise ValueError("selected task-worker pair is not currently feasible")
        workflow_id, task = decision.ref
        instance = self.instances[workflow_id]
        processor = decision.processor
        input_ready = _input_ready_time(
            instance,
            task,
            processor,
            self.status,
            self.executions,
        )
        if input_ready is None or input_ready > self.now + TOLERANCE:
            raise AssertionError("selected task-worker pair must be feasible")
        estimated_duration = instance.template.workflow.computation_cost(
            task,
            processor,
        )
        actual_duration = instance.actual_workflow.computation_cost(
            task,
            processor,
        )
        execution = DynamicTaskExecution(
            workflow_id=workflow_id,
            template_name=instance.template.name,
            task=task,
            source_id=instance.template.source_task_ids[task],
            program=instance.template.programs[task],
            processor=processor,
            start=self.now,
            finish=self.now + actual_duration,
            estimated_duration=estimated_duration,
            actual_duration=actual_duration,
            input_ready_time=input_ready,
        )
        self.executions[decision.ref] = execution
        self.status[decision.ref] = RUNNING
        self.running_by_processor[processor] = decision.ref
        self.committed_decisions += 1
        return execution

    def advance_to_next_event(
        self,
        policy: str | None = None,
    ) -> tuple[float, int]:
        """Advance the clock and return elapsed time and prior active count."""

        if self.is_complete:
            return 0.0, 0
        self.process_current_events()
        future_times: list[float] = []
        future_times.extend(
            execution.finish
            for ref, execution in self.executions.items()
            if self.status[ref] == RUNNING
            and execution.finish > self.now + TOLERANCE
        )
        future_times.extend(
            instance.arrival_time
            for instance in self.scenario.instances
            if instance.id not in self.arrived_ids
            and instance.arrival_time > self.now + TOLERANCE
        )
        next_data = _next_data_availability(
            self.now,
            self.scenario,
            self.arrived_ids,
            self.status,
            self.executions,
            self.running_by_processor,
            policy or ONLINE_GREEDY,
            self.static_assignment,
            self.static_orders,
        )
        if next_data is not None:
            future_times.append(next_data)
        if not future_times:
            pending = sorted(
                ref for ref, value in self.status.items() if value != COMPLETED
            )
            raise ValueError(
                f"dynamic simulation deadlocked at {self.now}: {pending}"
            )
        next_time = min(future_times)
        if next_time <= self.now + TOLERANCE:
            raise ValueError("dynamic event clock failed to advance")
        active_count = self.active_workflow_count
        elapsed = next_time - self.now
        self.now = next_time
        self._processed_event_time = None
        return elapsed, active_count

    def result(self, policy: str) -> DynamicSimulationResult:
        """Validate and summarize a completed execution."""

        if not self.is_complete:
            raise ValueError("cannot finalize an incomplete dynamic simulation")
        validation_errors = _validate_dynamic_schedule(
            self.scenario,
            self.executions,
        )
        workflow_results, metrics = _build_metrics(
            self.scenario,
            self.executions,
            self.scheduling_rounds,
            self.committed_decisions,
            self.candidate_evaluations,
            self.scheduler_wall_seconds,
        )
        ordered_tasks = tuple(
            sorted(
                self.executions.values(),
                key=lambda entry: (
                    entry.start,
                    entry.finish,
                    entry.processor,
                    entry.workflow_id,
                    entry.task,
                ),
            )
        )
        return DynamicSimulationResult(
            policy=policy,
            tasks=ordered_tasks,
            workflows=workflow_results,
            metrics=metrics,
            event_count=self.event_count,
            validation_errors=validation_errors,
        )


def simulate_dynamic_scenario(
    scenario: DynamicScenario,
    policy: str,
    aging_weight: float = 1.0,
) -> DynamicSimulationResult:
    """Run one online policy against an immutable dynamic scenario."""

    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown dynamic scheduling policy: {policy}")
    core = DynamicSchedulingCore(scenario, aging_weight=aging_weight)
    while not core.is_complete:
        core.process_current_events()
        while len(core.running_by_processor) < len(scenario.processors):
            decision = core.choose(policy)
            if decision is None:
                break
            core.commit(decision)
        if core.is_complete:
            break
        core.advance_to_next_event(policy)
    return core.result(policy)
