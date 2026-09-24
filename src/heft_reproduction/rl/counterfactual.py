"""Counterfactual heuristic rollouts for state-conditional supervision."""

from __future__ import annotations

from copy import deepcopy

from ..dynamic_models import ONLINE_GREEDY, POLICY_NAMES
from ..dynamic_policies import choose_candidate
from ..dynamic_simulator import DynamicSchedulingCore


def finish_core(
    core: DynamicSchedulingCore,
    policy: str = ONLINE_GREEDY,
) -> float:
    """Complete a cloned scheduling state and return total workflow JCT."""

    while not core.is_complete:
        core.process_current_events()
        while len(core.running_by_processor) < len(core.scenario.processors):
            decision = core.choose(policy)
            if decision is None:
                break
            core.commit(decision)
        if not core.is_complete:
            core.advance_to_next_event(policy)
    result = core.result(f"counterfactual-{policy}")
    return sum(workflow.jct for workflow in result.workflows)


def heuristic_counterfactuals(
    core: DynamicSchedulingCore,
    continuation_policy: str = ONLINE_GREEDY,
) -> dict[str, float]:
    """Evaluate each distinct heuristic proposal with a shared continuation."""

    candidates = core.candidates()
    outcomes: dict[str, float] = {}
    proposal_values: dict[object, float] = {}
    for policy in POLICY_NAMES:
        proposal, _ = choose_candidate(policy, candidates)
        if proposal is None:
            continue
        if proposal not in proposal_values:
            branch = deepcopy(core)
            branch.commit(proposal)
            proposal_values[proposal] = finish_core(branch, continuation_policy)
        outcomes[policy] = proposal_values[proposal]
    if not outcomes:
        raise ValueError("counterfactual state has no heuristic proposal")
    return outcomes
