"""Transparent online policy choices for the dynamic simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .dynamic_models import (
    AGING_ROLLING_HEFT,
    ONLINE_GREEDY,
    POLICY_NAMES,
    ROLLING_HEFT,
    SHORTEST_REMAINING_WORK,
    STATIC_HEFT,
    TaskRef,
)
from .models import ProcessorId


@dataclass(frozen=True)
class SchedulingCandidate:
    """One task-worker pair feasible at the current event time."""

    ref: TaskRef
    processor: ProcessorId
    processor_index: int
    workflow_arrival: float
    estimated_finish: float
    upward_rank: float
    normalized_upward_rank: float
    aging_score: float
    workflow_remaining_work: float
    static_allowed: bool
    static_absolute_planned_start: float


def choose_candidate(
    policy: str,
    candidates: tuple[SchedulingCandidate, ...],
) -> tuple[SchedulingCandidate | None, int]:
    """Choose one assignment and report deterministic search effort."""

    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown dynamic scheduling policy: {policy}")

    if policy == STATIC_HEFT:
        considered = tuple(
            candidate for candidate in candidates if candidate.static_allowed
        )
        if not considered:
            return None, 0
        return (
            min(
                considered,
                key=lambda candidate: (
                    candidate.static_absolute_planned_start,
                    candidate.workflow_arrival,
                    candidate.ref[0],
                    candidate.ref[1],
                    candidate.processor_index,
                ),
            ),
            len(considered),
        )

    if not candidates:
        return None, 0

    if policy == ONLINE_GREEDY:
        return (
            min(
                candidates,
                key=lambda candidate: (
                    candidate.estimated_finish,
                    candidate.workflow_arrival,
                    candidate.ref[0],
                    candidate.ref[1],
                    candidate.processor_index,
                ),
            ),
            len(candidates),
        )

    if policy == ROLLING_HEFT:
        return (
            min(
                candidates,
                key=lambda candidate: (
                    -round(candidate.upward_rank, 12),
                    candidate.workflow_arrival,
                    candidate.ref[0],
                    candidate.ref[1],
                    candidate.estimated_finish,
                    candidate.processor_index,
                ),
            ),
            len(candidates),
        )

    if policy == AGING_ROLLING_HEFT:
        return (
            min(
                candidates,
                key=lambda candidate: (
                    -round(candidate.aging_score, 12),
                    candidate.workflow_arrival,
                    candidate.ref[0],
                    candidate.ref[1],
                    candidate.estimated_finish,
                    candidate.processor_index,
                ),
            ),
            len(candidates),
        )

    if policy == SHORTEST_REMAINING_WORK:
        return (
            min(
                candidates,
                key=lambda candidate: (
                    round(candidate.workflow_remaining_work, 12),
                    -round(candidate.normalized_upward_rank, 12),
                    candidate.workflow_arrival,
                    candidate.ref[0],
                    candidate.ref[1],
                    candidate.estimated_finish,
                    candidate.processor_index,
                ),
            ),
            len(candidates),
        )

    raise AssertionError("all policy names must be handled")
