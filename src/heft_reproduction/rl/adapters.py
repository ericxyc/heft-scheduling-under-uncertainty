"""Mask-aware random and heuristic episode runners."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Any

from ..dynamic_models import POLICY_NAMES, DynamicSimulationResult
from ..dynamic_policies import choose_candidate
from .environment import DynamicSchedulingEnv


@dataclass(frozen=True)
class EpisodeOutcome:
    """Completed environment episode and its undiscounted rewards."""

    result: DynamicSimulationResult
    raw_return: float
    scaled_return: float
    truncated_candidates: int
    final_info: dict[str, object]


def _outcome(
    env: DynamicSchedulingEnv,
    raw_return: float,
    scaled_return: float,
    info: dict[str, object],
) -> EpisodeOutcome:
    result = info.get("result")
    if not isinstance(result, DynamicSimulationResult):
        raise AssertionError("completed episode must expose a simulation result")
    return EpisodeOutcome(
        result=result,
        raw_return=raw_return,
        scaled_return=scaled_return,
        truncated_candidates=int(info["truncated_candidates"]),
        final_info=dict(info),
    )


def run_random_episode(
    env: DynamicSchedulingEnv,
    scenario_seed: int,
    policy_seed: int,
) -> EpisodeOutcome:
    """Complete one episode by uniformly sampling valid action slots."""

    env.result_policy = "random-masked"
    env.reset(seed=scenario_seed)
    generator = Random(policy_seed)
    terminated = False
    raw_return = 0.0
    scaled_return = 0.0
    info: dict[str, object] = {}
    while not terminated:
        valid = [
            index
            for index, allowed in enumerate(env.action_masks())
            if allowed
        ]
        action = generator.choice(valid)
        _, reward, terminated, truncated, info = env.step(action)
        if truncated:
            raise AssertionError("finite scheduling episodes must not truncate")
        scaled_return += reward
        raw_return = float(info["raw_episode_return"])
    return _outcome(env, raw_return, scaled_return, info)


def run_heuristic_episode(
    env: DynamicSchedulingEnv,
    policy: str,
    scenario_seed: int,
) -> EpisodeOutcome:
    """Drive the Gym environment with an existing transparent heuristic."""

    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown dynamic scheduling policy: {policy}")
    env.result_policy = policy
    env.reset(seed=scenario_seed)
    terminated = False
    raw_return = 0.0
    scaled_return = 0.0
    info: dict[str, object] = {}
    while not terminated:
        started = perf_counter()
        decision, evaluated = choose_candidate(
            policy,
            tuple(env.current_candidates),
        )
        wall_seconds = perf_counter() - started
        if decision is None:
            transition = env.advance_for_policy(
                policy,
                candidate_evaluations=evaluated,
                decision_wall_seconds=wall_seconds,
            )
        else:
            action = tuple(env.current_candidates).index(decision)
            transition = env.step_with_effort(
                action,
                candidate_evaluations=evaluated,
                decision_wall_seconds=wall_seconds,
            )
        _, reward, terminated, truncated, info = transition
        if truncated:
            raise AssertionError("finite scheduling episodes must not truncate")
        scaled_return += reward
        raw_return = float(info["raw_episode_return"])
    return _outcome(env, raw_return, scaled_return, info)


def run_model_episode(
    env: DynamicSchedulingEnv,
    model: Any,
    scenario_seed: int,
    deterministic: bool = True,
) -> EpisodeOutcome:
    """Evaluate a MaskablePPO-compatible model with measured inference time."""

    env.result_policy = "maskable-ppo"
    observation, _ = env.reset(seed=scenario_seed)
    terminated = False
    raw_return = 0.0
    scaled_return = 0.0
    info: dict[str, object] = {}
    while not terminated:
        mask = env.action_masks()
        started = perf_counter()
        action, _ = model.predict(
            observation,
            action_masks=mask,
            deterministic=deterministic,
        )
        wall_seconds = perf_counter() - started
        observation, reward, terminated, truncated, info = (
            env.step_with_effort(
                int(action),
                candidate_evaluations=len(env.current_candidates),
                decision_wall_seconds=wall_seconds,
            )
        )
        if truncated:
            raise AssertionError("finite scheduling episodes must not truncate")
        scaled_return += reward
        raw_return = float(info["raw_episode_return"])
    return _outcome(env, raw_return, scaled_return, info)
