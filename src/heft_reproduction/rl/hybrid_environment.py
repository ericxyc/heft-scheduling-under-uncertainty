"""RL environment that chooses among transparent heuristic proposals."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np

from ..dynamic_models import POLICY_NAMES
from ..dynamic_policies import SchedulingCandidate, choose_candidate
from .environment import DynamicSchedulingEnv, ScenarioFactory
from .observation import GLOBAL_FEATURES, candidate_feature_count


class HeuristicSelectionEnv(gym.Env[dict[str, np.ndarray], int]):
    """Use RL to select one of the five Phase 4B policy recommendations."""

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
        self.base = DynamicSchedulingEnv(
            scenario_factory=scenario_factory,
            processors=processors,
            max_candidates=max_candidates,
            reward_scale=reward_scale,
            aging_weight=aging_weight,
        )
        self.processors = processors
        self.max_candidates = max_candidates
        self.action_space = gym.spaces.Discrete(len(POLICY_NAMES))
        self.observation_space = gym.spaces.Dict(
            {
                "proposals": gym.spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(
                        len(POLICY_NAMES),
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
                    shape=(len(POLICY_NAMES),),
                    dtype=np.int8,
                ),
            }
        )
        self._proposals: tuple[SchedulingCandidate | None, ...] = ()
        self._proposal_evaluations = 0
        self._proposal_wall_seconds = 0.0
        self._observation: dict[str, np.ndarray] | None = None
        self.result_policy = "hybrid-maskable-ppo"
        self.action_counts = [0 for _ in POLICY_NAMES]

    @property
    def core(self):
        return self.base.core

    @property
    def current_candidates(self) -> tuple[SchedulingCandidate | None, ...]:
        return self._proposals

    def _refresh(
        self,
        base_observation: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        started = perf_counter()
        candidates = tuple(self.base.current_candidates)
        proposals: list[SchedulingCandidate | None] = []
        evaluations = 0
        rows = np.zeros(
            self.observation_space["proposals"].shape,
            dtype=np.float32,
        )
        mask = np.zeros(len(POLICY_NAMES), dtype=np.int8)
        for index, policy in enumerate(POLICY_NAMES):
            proposal, evaluated = choose_candidate(policy, candidates)
            proposals.append(proposal)
            evaluations += evaluated
            if proposal is not None:
                base_index = candidates.index(proposal)
                rows[index] = base_observation["candidates"][base_index]
                mask[index] = 1
        if not mask.any():
            raise AssertionError("a scheduling decision needs a heuristic proposal")
        self._proposals = tuple(proposals)
        self._proposal_evaluations = evaluations
        self._proposal_wall_seconds = perf_counter() - started
        self._observation = {
            "proposals": rows,
            "global": base_observation["global"].copy(),
            "action_mask": mask,
        }
        return self._observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self.action_counts = [0 for _ in POLICY_NAMES]
        self.base.result_policy = self.result_policy
        observation, info = self.base.reset(seed=seed, options=options)
        return self._refresh(observation), info

    def action_masks(self) -> np.ndarray:
        if self._observation is None:
            return np.zeros(len(POLICY_NAMES), dtype=bool)
        return self._observation["action_mask"].astype(bool, copy=True)

    def step(self, action: int):
        return self.step_with_effort(
            action,
            candidate_evaluations=self._proposal_evaluations,
            decision_wall_seconds=0.0,
        )

    def step_with_effort(
        self,
        action: int,
        candidate_evaluations: int,
        decision_wall_seconds: float = 0.0,
    ):
        del candidate_evaluations
        index = int(action)
        mask = self.action_masks()
        if index < 0 or index >= len(POLICY_NAMES) or not mask[index]:
            raise ValueError(f"invalid or masked heuristic action: {action}")
        proposal = self._proposals[index]
        if proposal is None:
            raise AssertionError("valid heuristic action must have a proposal")
        self.action_counts[index] += 1
        base_action = tuple(self.base.current_candidates).index(proposal)
        transition = self.base.step_with_effort(
            base_action,
            candidate_evaluations=self._proposal_evaluations,
            decision_wall_seconds=(
                decision_wall_seconds + self._proposal_wall_seconds
            ),
        )
        observation, reward, terminated, truncated, info = transition
        info["heuristic_action_counts"] = {
            policy: self.action_counts[action_index]
            for action_index, policy in enumerate(POLICY_NAMES)
        }
        if terminated:
            self._observation = None
            self._proposals = ()
            return (
                {
                    "proposals": np.zeros(
                        self.observation_space["proposals"].shape,
                        dtype=np.float32,
                    ),
                    "global": np.zeros(
                        self.observation_space["global"].shape,
                        dtype=np.float32,
                    ),
                    "action_mask": np.zeros(
                        len(POLICY_NAMES),
                        dtype=np.int8,
                    ),
                },
                reward,
                terminated,
                truncated,
                info,
            )
        return (
            self._refresh(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    @property
    def final_result(self):
        return self.base.final_result
