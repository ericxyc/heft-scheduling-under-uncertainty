"""Multi-seed, multi-load benchmark sweeps for dynamic scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import fmean, stdev
from typing import Mapping, Sequence

from .benchmark_data import BenchmarkCorpus
from .dynamic_models import (
    POLICY_NAMES,
    DynamicMetrics,
    WorkflowTemplate,
)
from .dynamic_scenario import build_dynamic_scenario
from .dynamic_simulator import simulate_dynamic_scenario


AGGREGATE_METRICS = (
    "mean_jct",
    "p95_jct",
    "mean_response_time",
    "mean_task_queue_wait",
    "p95_task_queue_wait",
    "throughput_workflows_per_second",
    "average_utilization",
    "simulation_horizon",
    "cross_worker_communication_seconds",
    "candidate_evaluations",
    "scheduler_wall_seconds",
)


@dataclass(frozen=True)
class BenchmarkRun:
    """One policy evaluated on one seeded dynamic scenario."""

    size: str
    offered_load: float
    runtime_cv: float
    replicate: int
    scenario_seed: int
    mean_interarrival_time: float
    policy: str
    metrics: DynamicMetrics
    is_valid: bool
    validation_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "offered_load": self.offered_load,
            "runtime_cv": self.runtime_cv,
            "replicate": self.replicate,
            "scenario_seed": self.scenario_seed,
            "mean_interarrival_time": self.mean_interarrival_time,
            "policy": self.policy,
            "metrics": self.metrics.to_dict(),
            "is_valid": self.is_valid,
            "validation_errors": list(self.validation_errors),
        }


@dataclass(frozen=True)
class AggregateStatistic:
    """Sample mean, standard deviation, and normal CI for one metric."""

    mean: float
    sample_std: float
    ci95_half_width: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "sample_std": self.sample_std,
            "ci95_half_width": self.ci95_half_width,
        }


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Aggregate policy result for one size/load/CV cell."""

    size: str
    offered_load: float
    runtime_cv: float
    policy: str
    replicate_count: int
    statistics: Mapping[str, AggregateStatistic]

    def to_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "offered_load": self.offered_load,
            "runtime_cv": self.runtime_cv,
            "policy": self.policy,
            "replicate_count": self.replicate_count,
            "statistics": {
                name: statistic.to_dict()
                for name, statistic in self.statistics.items()
            },
        }


@dataclass(frozen=True)
class DynamicBenchmarkResult:
    """Complete Phase 4B benchmark output."""

    workflow_count_per_scenario: int
    replicate_count: int
    base_seed: int
    aging_weight: float
    sizes: tuple[str, ...]
    loads: tuple[float, ...]
    cvs: tuple[float, ...]
    policies: tuple[str, ...]
    reference_work_by_size: Mapping[str, float]
    runs: tuple[BenchmarkRun, ...]
    aggregates: tuple[BenchmarkAggregate, ...]

    @property
    def is_valid(self) -> bool:
        return all(run.is_valid for run in self.runs)

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_count_per_scenario": self.workflow_count_per_scenario,
            "replicate_count": self.replicate_count,
            "base_seed": self.base_seed,
            "aging_weight": self.aging_weight,
            "sizes": list(self.sizes),
            "loads": list(self.loads),
            "cvs": list(self.cvs),
            "policies": list(self.policies),
            "reference_work_by_size": dict(self.reference_work_by_size),
            "runs": [run.to_dict() for run in self.runs],
            "aggregates": [
                aggregate.to_dict() for aggregate in self.aggregates
            ],
            "is_valid": self.is_valid,
        }


def reference_work(
    templates: Sequence[WorkflowTemplate],
    workflow_count: int,
) -> float:
    """Mean estimated worker-seconds for the round-robin template mix."""

    if not templates:
        raise ValueError("at least one template is required")
    if workflow_count <= 0:
        raise ValueError("workflow count must be positive")
    selected = [
        templates[index % len(templates)]
        for index in range(workflow_count)
    ]
    return fmean(
        sum(
            template.workflow.mean_computation_cost(task)
            for task in template.workflow.tasks
        )
        for template in selected
    )


def mean_interarrival_for_load(
    templates: Sequence[WorkflowTemplate],
    workflow_count: int,
    offered_load: float,
) -> float:
    """Convert a normalized offered load into a mean inter-arrival time."""

    if not isfinite(offered_load) or offered_load <= 0:
        raise ValueError("offered load must be positive and finite")
    work = reference_work(templates, workflow_count)
    return work / (len(templates[0].workflow.processors) * offered_load)


def _metric_value(metrics: DynamicMetrics, name: str) -> float:
    value = getattr(metrics, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"dynamic metric {name} must be numeric")
    return float(value)


def _aggregate_runs(runs: Sequence[BenchmarkRun]) -> BenchmarkAggregate:
    if not runs:
        raise ValueError("cannot aggregate an empty run group")
    first = runs[0]
    statistics: dict[str, AggregateStatistic] = {}
    for metric_name in AGGREGATE_METRICS:
        values = [_metric_value(run.metrics, metric_name) for run in runs]
        sample_std = stdev(values) if len(values) > 1 else 0.0
        statistics[metric_name] = AggregateStatistic(
            mean=fmean(values),
            sample_std=sample_std,
            ci95_half_width=1.96 * sample_std / sqrt(len(values)),
        )
    return BenchmarkAggregate(
        size=first.size,
        offered_load=first.offered_load,
        runtime_cv=first.runtime_cv,
        policy=first.policy,
        replicate_count=len(runs),
        statistics=statistics,
    )


def run_dynamic_benchmark(
    corpus: BenchmarkCorpus,
    sizes: Sequence[str],
    loads: Sequence[float],
    cvs: Sequence[float],
    replicate_count: int,
    workflow_count: int,
    base_seed: int,
    policies: Sequence[str] = POLICY_NAMES,
    aging_weight: float = 1.0,
) -> DynamicBenchmarkResult:
    """Evaluate every policy on shared seeded scenarios and aggregate results."""

    selected_sizes = tuple(sizes)
    selected_loads = tuple(float(load) for load in loads)
    selected_cvs = tuple(float(cv) for cv in cvs)
    selected_policies = tuple(policies)
    if not selected_sizes:
        raise ValueError("at least one benchmark size is required")
    if any(size not in corpus.sizes for size in selected_sizes):
        raise ValueError(f"unknown benchmark size; choices are {corpus.sizes}")
    if not selected_loads or any(
        not isfinite(load) or load <= 0 for load in selected_loads
    ):
        raise ValueError("loads must be positive and finite")
    if not selected_cvs or any(
        not isfinite(cv) or cv < 0 for cv in selected_cvs
    ):
        raise ValueError("CV values must be finite and non-negative")
    if replicate_count <= 0:
        raise ValueError("replicate count must be positive")
    if workflow_count <= 0:
        raise ValueError("workflow count must be positive")
    if len(set(selected_policies)) != len(selected_policies):
        raise ValueError("benchmark policy names must be unique")
    if any(policy not in POLICY_NAMES for policy in selected_policies):
        raise ValueError(f"unknown policy; choices are {POLICY_NAMES}")
    if not isfinite(aging_weight) or aging_weight < 0:
        raise ValueError("aging weight must be finite and non-negative")

    all_runs: list[BenchmarkRun] = []
    reference_by_size: dict[str, float] = {}
    for size_index, size in enumerate(selected_sizes):
        templates = corpus.templates_for_size(size)
        reference_by_size[size] = reference_work(
            templates,
            workflow_count,
        )
        for load_index, offered_load in enumerate(selected_loads):
            mean_interarrival = mean_interarrival_for_load(
                templates,
                workflow_count,
                offered_load,
            )
            for cv_index, runtime_cv in enumerate(selected_cvs):
                for replicate in range(replicate_count):
                    scenario_seed = (
                        base_seed
                        + size_index * 10_000_000
                        + load_index * 1_000_000
                        + cv_index * 100_000
                        + replicate
                    )
                    scenario = build_dynamic_scenario(
                        templates=templates,
                        workflow_count=workflow_count,
                        mean_interarrival_time=mean_interarrival,
                        runtime_cv=runtime_cv,
                        seed=scenario_seed,
                    )
                    for policy in selected_policies:
                        result = simulate_dynamic_scenario(
                            scenario,
                            policy,
                            aging_weight=aging_weight,
                        )
                        all_runs.append(
                            BenchmarkRun(
                                size=size,
                                offered_load=offered_load,
                                runtime_cv=runtime_cv,
                                replicate=replicate,
                                scenario_seed=scenario_seed,
                                mean_interarrival_time=mean_interarrival,
                                policy=policy,
                                metrics=result.metrics,
                                is_valid=result.is_valid,
                                validation_errors=result.validation_errors,
                            )
                        )

    grouped: dict[
        tuple[str, float, float, str],
        list[BenchmarkRun],
    ] = {}
    for run in all_runs:
        key = (run.size, run.offered_load, run.runtime_cv, run.policy)
        grouped.setdefault(key, []).append(run)
    aggregates = tuple(
        _aggregate_runs(grouped[key])
        for key in sorted(
            grouped,
            key=lambda value: (
                selected_sizes.index(value[0]),
                selected_loads.index(value[1]),
                selected_cvs.index(value[2]),
                selected_policies.index(value[3]),
            ),
        )
    )

    return DynamicBenchmarkResult(
        workflow_count_per_scenario=workflow_count,
        replicate_count=replicate_count,
        base_seed=base_seed,
        aging_weight=aging_weight,
        sizes=selected_sizes,
        loads=selected_loads,
        cvs=selected_cvs,
        policies=selected_policies,
        reference_work_by_size=reference_by_size,
        runs=tuple(all_runs),
        aggregates=aggregates,
    )
