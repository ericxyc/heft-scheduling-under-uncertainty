"""Gymnasium environment backed by the dynamic scheduling event core."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np

from ..dynamic_models import DynamicScenario, DynamicSimulationResult
from ..dynamic_simulator import DynamicSchedulingCore
from .observation import (
    GLOBAL_FEATURES,
    EncodedObservation,
    candidate_feature_count,
    encode_observation,
)


ScenarioFactory = Callable[[int], DynamicScenario]


class DynamicSchedulingEnv(gym.Env[dict[str, np.ndarray], int]):
    """Select one feasible task-worker pair at each scheduling decision."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_factory: ScenarioFactory,
        processors: tuple[str, ...],
        max_candidates: int = 128,
        reward_scale: float = 1.0,
        aging_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not processors:
            raise ValueError("processors must be non-empty")
        if max_candidates <= 0:
            raise ValueError("max candidates must be positive")
        if reward_scale <= 0:
            raise ValueError("reward scale must be positive")
        self.scenario_factory = scenario_factory
        self.processors = processors
        self.max_candidates = max_candidates
        self.reward_scale = reward_scale
        self.aging_weight = aging_weight
        self.action_space = gym.spaces.Discrete(max_candidates)
        self.observation_space = gym.spaces.Dict(
            {
                "candidates": gym.spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(
                        max_candidates,
                        candidate_feature_count(len(processors)),
                    ),
                    dtype=np.float32,
                ),
                "global": gym.spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(GLOBAL_FEATURES,),
                    dtype=np.float32,
                ),
                "action_mask": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(max_candidates,),
                    dtype=np.int8,
                ),
            }
        )
        self.core: DynamicSchedulingCore | None = None
        self._encoded: EncodedObservation | None = None
        self._raw_episode_return = 0.0
        self._truncation_count = 0
        self._scenario_seed = 0
        self._terminated = False
        self.result_policy = "rl-policy"

    def _require_core(self) -> DynamicSchedulingCore:
        if self.core is None:
            raise RuntimeError("environment must be reset before use")
        return self.core

    def _empty_observation(self) -> dict[str, np.ndarray]:
        return {
            "candidates": np.zeros(
                self.observation_space["candidates"].shape,
                dtype=np.float32,
            ),
            "global": np.zeros(
                self.observation_space["global"].shape,
                dtype=np.float32,
            ),
            "action_mask": np.zeros(
                self.max_candidates,
                dtype=np.int8,
            ),
        }

    def _advance_until_decision(self) -> float:
        core = self._require_core()
        raw_reward = 0.0
        while True:
            core.process_current_events()
            if core.is_complete:
                self._encoded = None
                return raw_reward
            encoded = encode_observation(core, self.max_candidates)
            if encoded.slots:
                self._encoded = encoded
                self._truncation_count += encoded.truncated_count
                return raw_reward
            elapsed, active_count = core.advance_to_next_event()
            raw_reward -= active_count * elapsed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._scenario_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, 2**31 - 1))
        )
        scenario = self.scenario_factory(self._scenario_seed)
        if scenario.processors != self.processors:
            raise ValueError(
                "scenario processors do not match the environment declaration"
            )
        self.core = DynamicSchedulingCore(
            scenario,
            aging_weight=self.aging_weight,
        )
        self._encoded = None
        self._raw_episode_return = 0.0
        self._truncation_count = 0
        self._terminated = False
        initial_reward = self._advance_until_decision()
        if initial_reward != 0.0:
            raise AssertionError("pre-arrival time must not accrue JCT reward")
        observation = (
            self._encoded.values
            if self._encoded is not None
            else self._empty_observation()
        )
        return observation, self._info()

    def action_masks(self) -> np.ndarray:
        """Return the valid-action mask expected by MaskablePPO."""

        if self._encoded is None:
            return np.zeros(self.max_candidates, dtype=bool)
        return self._encoded.values["action_mask"].astype(bool, copy=True)

    @property
    def current_candidates(self) -> tuple[Any, ...]:
        return self._encoded.slots if self._encoded is not None else ()

    def _info(self) -> dict[str, Any]:
        core = self._require_core()
        return {
            "scenario_seed": self._scenario_seed,
            "simulated_time": core.now,
            "active_workflows": core.active_workflow_count,
            "raw_candidate_count": (
                self._encoded.raw_candidate_count
                if self._encoded is not None
                else 0
            ),
            "truncated_candidates": self._truncation_count,
            "raw_episode_return": self._raw_episode_return,
        }

    def step(
        self,
        action: int,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        decision_start = perf_counter()
        result = self.step_with_effort(
            action,
            candidate_evaluations=(
                self._encoded.raw_candidate_count
                if self._encoded is not None
                else 0
            ),
            decision_wall_seconds=perf_counter() - decision_start,
        )
        return result

    def _transition_result(
        self,
        raw_reward: float,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        core = self._require_core()
        self._raw_episode_return += raw_reward
        self._terminated = core.is_complete
        info = self._info()
        if self._terminated:
            result = core.result(self.result_policy)
            total_jct = sum(workflow.jct for workflow in result.workflows)
            identity_error = self._raw_episode_return + total_jct
            info.update(
                {
                    "result": result,
                    "total_jct": total_jct,
                    "reward_identity_error": identity_error,
                }
            )
            observation = self._empty_observation()
        else:
            if self._encoded is None:
                raise AssertionError("non-terminal transition needs candidates")
            observation = self._encoded.values
        return (
            observation,
            raw_reward / self.reward_scale,
            self._terminated,
            False,
            info,
        )

    def step_with_effort(
        self,
        action: int,
        candidate_evaluations: int,
        decision_wall_seconds: float = 0.0,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Apply an action with externally measured policy effort."""

        if self._terminated:
            raise RuntimeError("cannot step a terminated environment")
        core = self._require_core()
        if self._encoded is None:
            raise RuntimeError("environment has no scheduling decision")
        action_index = int(action)
        mask = self.action_masks()
        if (
            action_index < 0
            or action_index >= self.max_candidates
            or not mask[action_index]
        ):
            raise ValueError(f"invalid or masked scheduling action: {action}")

        decision = self._encoded.slots[action_index]
        core.record_external_decision(
            candidate_evaluations,
            decision_wall_seconds,
        )
        core.commit(decision)
        raw_reward = self._advance_until_decision()
        return self._transition_result(raw_reward)

    def advance_for_policy(
        self,
        policy: str,
        candidate_evaluations: int,
        decision_wall_seconds: float = 0.0,
    ) -> tuple[
        dict[str, np.ndarray],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """Advance after a heuristic intentionally finds no legal assignment."""

        if self._terminated:
            raise RuntimeError("cannot advance a terminated environment")
        core = self._require_core()
        core.record_external_decision(
            candidate_evaluations,
            decision_wall_seconds,
        )
        self._encoded = None
        elapsed, active_count = core.advance_to_next_event(policy)
        raw_reward = -active_count * elapsed
        raw_reward += self._advance_until_decision()
        return self._transition_result(raw_reward)

    @property
    def final_result(self) -> DynamicSimulationResult:
        core = self._require_core()
        if not self._terminated:
            raise RuntimeError("episode has not completed")
        return core.result(self.result_policy)
