"""Controlled runtime-uncertainty experiments for fixed HEFT plans."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt
from random import Random
from statistics import fmean, stdev
from typing import Mapping, Sequence

from .models import ProcessorId, Schedule, ScheduleEntry, TaskId, Workflow
from .scheduler import calculate_makespan, schedule_heft, validate_schedule


@dataclass(frozen=True)
class UncertaintyTrial:
    """One sampled realization and the two schedules evaluated on it."""

    cv: float
    trial_index: int
    trial_seed: int
    static_plan_makespan: float
    perfect_information_heft_makespan: float

    @property
    def static_plan_gap(self) -> float:
        return self.static_plan_makespan - self.perfect_information_heft_makespan

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cv": self.cv,
            "trial_index": self.trial_index,
            "trial_seed": self.trial_seed,
            "static_plan_makespan": self.static_plan_makespan,
            "perfect_information_heft_makespan": (
                self.perfect_information_heft_makespan
            ),
            "static_plan_gap": self.static_plan_gap,
        }


@dataclass(frozen=True)
class UncertaintySummary:
    """Aggregate results for one uncertainty strength."""

    cv: float
    trial_count: int
    static_plan_mean_makespan: float
    static_plan_std_makespan: float
    static_plan_ci95_half_width: float
    perfect_information_heft_mean_makespan: float
    mean_static_plan_gap: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cv": self.cv,
            "trial_count": self.trial_count,
            "static_plan_mean_makespan": self.static_plan_mean_makespan,
            "static_plan_std_makespan": self.static_plan_std_makespan,
            "static_plan_ci95_half_width": self.static_plan_ci95_half_width,
            "perfect_information_heft_mean_makespan": (
                self.perfect_information_heft_mean_makespan
            ),
            "mean_static_plan_gap": self.mean_static_plan_gap,
        }


@dataclass(frozen=True)
class UncertaintyExperiment:
    """Complete, reproducible result of a fixed-plan uncertainty study."""

    planned_makespan: float
    distribution: str
    master_seed: int
    trials: tuple[UncertaintyTrial, ...]
    summaries: tuple[UncertaintySummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "planned_makespan": self.planned_makespan,
            "distribution": self.distribution,
            "master_seed": self.master_seed,
            "trials": [trial.to_dict() for trial in self.trials],
            "summaries": [summary.to_dict() for summary in self.summaries],
        }


def sample_duration_multipliers(
    tasks: Sequence[TaskId],
    cv: float,
    seed: int,
) -> dict[TaskId, float]:
    """Sample mean-one task multipliers from a seeded log-normal model."""

    if not isfinite(cv) or cv < 0:
        raise ValueError("CV must be finite and non-negative")
    if cv == 0:
        return {task: 1.0 for task in tasks}

    sigma = sqrt(log(1.0 + cv * cv))
    mu = -(sigma * sigma) / 2.0
    generator = Random(seed)
    return {
        task: generator.lognormvariate(mu, sigma)
        for task in tasks
    }


def workflow_with_actual_durations(
    workflow: Workflow,
    multipliers: Mapping[TaskId, float],
) -> Workflow:
    """Return the same DAG with task-level realized duration multipliers."""

    if set(multipliers) != set(workflow.tasks):
        raise ValueError("duration multipliers must cover every task exactly")
    if any(multiplier <= 0 for multiplier in multipliers.values()):
        raise ValueError("duration multipliers must be positive")

    return Workflow(
        tasks=workflow.tasks,
        processors=workflow.processors,
        computation_costs={
            task: {
                processor: workflow.computation_cost(task, processor)
                * multipliers[task]
                for processor in workflow.processors
            }
            for task in workflow.tasks
        },
        communication_costs=workflow.communication_costs,
    )


def replay_fixed_plan(
    estimated_workflow: Workflow,
    planned_schedule: Schedule,
    actual_workflow: Workflow,
) -> dict[TaskId, ScheduleEntry]:
    """Execute a fixed assignment/order plan with realized task durations."""

    if estimated_workflow.tasks != actual_workflow.tasks:
        raise ValueError("estimated and actual workflows must contain the same tasks")
    if estimated_workflow.processors != actual_workflow.processors:
        raise ValueError("estimated and actual workflows must contain the same workers")
    if estimated_workflow.communication_costs != actual_workflow.communication_costs:
        raise ValueError("estimated and actual workflows must have the same dependencies")
    planned_errors = validate_schedule(estimated_workflow, planned_schedule)
    if planned_errors:
        raise ValueError(f"planned schedule is invalid: {planned_errors}")

    worker_orders: dict[ProcessorId, list[TaskId]] = {
        processor: [] for processor in estimated_workflow.processors
    }
    for processor in estimated_workflow.processors:
        worker_orders[processor] = [
            entry.task
            for entry in sorted(
                (
                    entry
                    for entry in planned_schedule.values()
                    if entry.processor == processor
                ),
                key=lambda entry: (entry.start, entry.finish, entry.task),
            )
        ]

    positions = {processor: 0 for processor in estimated_workflow.processors}
    replay: dict[TaskId, ScheduleEntry] = {}
    while len(replay) < len(estimated_workflow.tasks):
        progressed = False
        for processor in estimated_workflow.processors:
            position = positions[processor]
            order = worker_orders[processor]
            if position >= len(order):
                continue
            task = order[position]
            parents = actual_workflow.predecessors(task)
            if any(parent not in replay for parent in parents):
                continue

            previous_finish = 0.0
            if position:
                previous_finish = replay[order[position - 1]].finish
            dependency_finish = max(
                (
                    replay[parent].finish
                    + (
                        0.0
                        if replay[parent].processor == processor
                        else actual_workflow.communication_cost(parent, task)
                    )
                    for parent in parents
                ),
                default=0.0,
            )
            start = max(previous_finish, dependency_finish)
            replay[task] = ScheduleEntry(
                task=task,
                processor=processor,
                start=start,
                finish=start + actual_workflow.computation_cost(task, processor),
            )
            positions[processor] += 1
            progressed = True
        if not progressed:
            raise ValueError("fixed worker orders cannot be replayed without a cycle")

    errors = validate_schedule(actual_workflow, replay)
    if errors:
        raise ValueError(f"replayed schedule is invalid: {errors}")
    return replay


def run_uncertainty_experiment(
    workflow: Workflow,
    planned_schedule: Schedule,
    cvs: Sequence[float],
    trials_per_cv: int,
    seed: int,
) -> UncertaintyExperiment:
    """Evaluate a fixed HEFT plan over seeded runtime-uncertainty trials."""

    if not cvs:
        raise ValueError("at least one CV value is required")
    if any(not isfinite(cv) or cv < 0 for cv in cvs):
        raise ValueError("CV values must be finite and non-negative")
    if trials_per_cv <= 0:
        raise ValueError("trials per CV must be positive")
    planned_errors = validate_schedule(workflow, planned_schedule)
    if planned_errors:
        raise ValueError(f"planned schedule is invalid: {planned_errors}")

    master = Random(seed)
    trial_results: list[UncertaintyTrial] = []
    summaries: list[UncertaintySummary] = []
    planned_makespan = calculate_makespan(planned_schedule)
    for cv in cvs:
        cv_trials: list[UncertaintyTrial] = []
        for trial_index in range(trials_per_cv):
            trial_seed = master.randrange(0, 2**63)
            multipliers = sample_duration_multipliers(
                workflow.tasks,
                cv,
                trial_seed,
            )
            actual_workflow = workflow_with_actual_durations(
                workflow,
                multipliers,
            )
            static_replay = replay_fixed_plan(
                workflow,
                planned_schedule,
                actual_workflow,
            )
            perfect_information = schedule_heft(actual_workflow)
            if not perfect_information.is_valid:
                raise ValueError(
                    "perfect-information HEFT produced an invalid schedule: "
                    f"{perfect_information.validation_errors}"
                )
            trial = UncertaintyTrial(
                cv=float(cv),
                trial_index=trial_index,
                trial_seed=trial_seed,
                static_plan_makespan=calculate_makespan(static_replay),
                perfect_information_heft_makespan=perfect_information.makespan,
            )
            cv_trials.append(trial)
            trial_results.append(trial)

        static_makespans = [trial.static_plan_makespan for trial in cv_trials]
        perfect_makespans = [
            trial.perfect_information_heft_makespan for trial in cv_trials
        ]
        gaps = [trial.static_plan_gap for trial in cv_trials]
        static_std = stdev(static_makespans) if len(static_makespans) > 1 else 0.0
        summaries.append(
            UncertaintySummary(
                cv=float(cv),
                trial_count=len(cv_trials),
                static_plan_mean_makespan=fmean(static_makespans),
                static_plan_std_makespan=static_std,
                static_plan_ci95_half_width=(
                    1.96 * static_std / sqrt(len(static_makespans))
                ),
                perfect_information_heft_mean_makespan=fmean(perfect_makespans),
                mean_static_plan_gap=fmean(gaps),
            )
        )

    return UncertaintyExperiment(
        planned_makespan=planned_makespan,
        distribution="task-level mean-one log-normal multipliers",
        master_seed=seed,
        trials=tuple(trial_results),
        summaries=tuple(summaries),
    )
