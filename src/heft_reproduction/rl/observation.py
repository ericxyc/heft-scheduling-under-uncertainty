"""Fixed-shape, non-oracle observations for dynamic scheduling agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..dynamic_policies import SchedulingCandidate
from ..dynamic_simulator import COMPLETED, DynamicSchedulingCore


BASE_CANDIDATE_FEATURES = 10
GLOBAL_FEATURES = 9


@dataclass(frozen=True)
class EncodedObservation:
    """One observation and its stable action-slot mapping."""

    values: dict[str, np.ndarray]
    slots: tuple[SchedulingCandidate, ...]
    raw_candidate_count: int
    truncated_count: int


def candidate_feature_count(processor_count: int) -> int:
    if processor_count <= 0:
        raise ValueError("processor count must be positive")
    return BASE_CANDIDATE_FEATURES + processor_count


def _stable_key(
    candidate: SchedulingCandidate,
) -> tuple[str, int, int]:
    return (
        candidate.ref[0],
        candidate.ref[1],
        candidate.processor_index,
    )


def _diverse_shortlist(
    candidates: Iterable[SchedulingCandidate],
    limit: int,
) -> tuple[SchedulingCandidate, ...]:
    """Interleave strong transparent rankings before deterministic fallback."""

    values = tuple(candidates)
    if limit <= 0:
        raise ValueError("candidate limit must be positive")
    if len(values) <= limit:
        return tuple(sorted(values, key=_stable_key))

    orderings = (
        sorted(
            values,
            key=lambda item: (item.estimated_finish, *_stable_key(item)),
        ),
        sorted(
            values,
            key=lambda item: (
                -item.normalized_upward_rank,
                *_stable_key(item),
            ),
        ),
        sorted(
            values,
            key=lambda item: (-item.aging_score, *_stable_key(item)),
        ),
        sorted(
            values,
            key=lambda item: (
                item.workflow_remaining_work,
                *_stable_key(item),
            ),
        ),
        sorted(
            values,
            key=lambda item: (
                not item.static_allowed,
                item.static_absolute_planned_start,
                *_stable_key(item),
            ),
        ),
    )
    chosen: list[SchedulingCandidate] = []
    seen: set[tuple[tuple[str, int], str]] = set()
    index = 0
    while len(chosen) < limit:
        added = False
        for ordering in orderings:
            if index >= len(ordering):
                continue
            candidate = ordering[index]
            key = (candidate.ref, candidate.processor)
            if key not in seen:
                seen.add(key)
                chosen.append(candidate)
                added = True
                if len(chosen) == limit:
                    break
        index += 1
        if not added and index >= len(values):
            break
    return tuple(chosen)


def _scales(core: DynamicSchedulingCore) -> tuple[float, float, int]:
    time_scale = max(core.baseline_makespans.values())
    work_scale = max(
        sum(
            instance.template.workflow.mean_computation_cost(task)
            for task in instance.template.workflow.tasks
        )
        for instance in core.scenario.instances
    )
    max_tasks = max(
        len(instance.template.workflow.tasks)
        for instance in core.scenario.instances
    )
    return max(time_scale, 1e-9), max(work_scale, 1e-9), max_tasks


def _candidate_row(
    core: DynamicSchedulingCore,
    candidate: SchedulingCandidate,
    time_scale: float,
    work_scale: float,
    max_tasks: int,
) -> np.ndarray:
    workflow_id, task = candidate.ref
    instance = core.instances[workflow_id]
    workflow = instance.template.workflow
    duration = candidate.estimated_finish - core.now
    mean_duration = workflow.mean_computation_cost(task)
    completed = sum(
        core.status[instance.ref(item)] == COMPLETED
        for item in workflow.tasks
    )
    incoming = sum(
        workflow.communication_cost(parent, task)
        for parent in workflow.predecessors(task)
    )
    outgoing = sum(
        workflow.communication_cost(task, child)
        for child in workflow.successors(task)
    )
    maximum_rank = max(core.ranks[workflow_id].values())
    normalized_age = (
        max(0.0, core.now - instance.arrival_time)
        / core.baseline_makespans[workflow_id]
    )
    row = [
        duration / time_scale,
        duration / max(mean_duration, 1e-9),
        core.ranks[workflow_id][task] / maximum_rank,
        normalized_age,
        candidate.workflow_remaining_work / work_scale,
        completed / len(workflow.tasks),
        len(workflow.successors(task)) / max_tasks,
        incoming / time_scale,
        outgoing / time_scale,
        float(candidate.static_allowed),
    ]
    row.extend(
        float(index == candidate.processor_index)
        for index in range(len(core.scenario.processors))
    )
    return np.clip(np.asarray(row, dtype=np.float32), 0.0, 10.0)


def encode_observation(
    core: DynamicSchedulingCore,
    max_candidates: int,
) -> EncodedObservation:
    """Encode current feasible assignments without realized future durations."""

    raw_candidates = core.candidates()
    slots = _diverse_shortlist(raw_candidates, max_candidates)
    processor_count = len(core.scenario.processors)
    feature_count = candidate_feature_count(processor_count)
    candidate_features = np.zeros(
        (max_candidates, feature_count),
        dtype=np.float32,
    )
    mask = np.zeros(max_candidates, dtype=np.int8)
    time_scale, work_scale, max_tasks = _scales(core)
    for index, candidate in enumerate(slots):
        candidate_features[index] = _candidate_row(
            core,
            candidate,
            time_scale,
            work_scale,
            max_tasks,
        )
        mask[index] = 1

    total_tasks = len(core.status)
    completed_tasks = sum(value == COMPLETED for value in core.status.values())
    workflow_count = len(core.scenario.instances)
    truncated_count = len(raw_candidates) - len(slots)
    global_features = np.clip(
        np.asarray(
            [
                core.now / time_scale,
                core.active_workflow_count / workflow_count,
                len(core.arrived_ids) / workflow_count,
                completed_tasks / total_tasks,
                len(core.running_by_processor) / processor_count,
                len(raw_candidates) / max_candidates,
                core.scenario.runtime_cv,
                core.scenario.mean_interarrival_time / time_scale,
                truncated_count / max(len(raw_candidates), 1),
            ],
            dtype=np.float32,
        ),
        0.0,
        10.0,
    )
    return EncodedObservation(
        values={
            "candidates": candidate_features,
            "global": global_features,
            "action_mask": mask,
        },
        slots=slots,
        raw_candidate_count=len(raw_candidates),
        truncated_count=truncated_count,
    )
