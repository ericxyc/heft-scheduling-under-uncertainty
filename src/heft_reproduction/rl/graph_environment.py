"""Dynamic scheduling environment with a fixed-shape DAG observation."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ..dynamic_models import POLICY_NAMES
from ..dynamic_policies import choose_candidate
from ..dynamic_simulator import DynamicSchedulingCore
from .environment import DynamicSchedulingEnv
from .graph_observation import (
    GRAPH_EDGE_FEATURES,
    GRAPH_NODE_FEATURES,
    encode_graph_observation,
)
from .observation import GLOBAL_FEATURES, candidate_feature_count, encode_observation


class GraphSchedulingEnv(DynamicSchedulingEnv):
    """Select task-worker candidates using candidate and DAG context."""

    def __init__(
        self,
        *args,
        max_nodes: int = 2048,
        max_edges: int = 4096,
        **kwargs,
    ) -> None:
        if max_nodes <= 0 or max_edges <= 0:
            raise ValueError("graph limits must be positive")
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        super().__init__(*args, **kwargs)
        processor_count = len(self.processors)
        self.observation_space = gym.spaces.Dict(
            {
                "candidates": gym.spaces.Box(
                    0.0,
                    10.0,
                    shape=(
                        self.max_candidates,
                        candidate_feature_count(processor_count),
                    ),
                    dtype=np.float32,
                ),
                "global": gym.spaces.Box(
                    0.0, 10.0, shape=(GLOBAL_FEATURES,), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(
                    0, 1, shape=(self.max_candidates,), dtype=np.int8
                ),
                "graph_nodes": gym.spaces.Box(
                    0.0,
                    10.0,
                    shape=(max_nodes, GRAPH_NODE_FEATURES),
                    dtype=np.float32,
                ),
                "graph_node_mask": gym.spaces.Box(
                    0, 1, shape=(max_nodes,), dtype=np.int8
                ),
                "graph_edge_index": gym.spaces.Box(
                    0,
                    max_nodes - 1,
                    shape=(max_edges, 2),
                    dtype=np.int32,
                ),
                "graph_edge_features": gym.spaces.Box(
                    0.0,
                    10.0,
                    shape=(max_edges, GRAPH_EDGE_FEATURES),
                    dtype=np.float32,
                ),
                "graph_edge_mask": gym.spaces.Box(
                    0, 1, shape=(max_edges,), dtype=np.int8
                ),
                "candidate_nodes": gym.spaces.Box(
                    0,
                    max_nodes - 1,
                    shape=(self.max_candidates,),
                    dtype=np.int32,
                ),
            }
        )

    def _encode_observation(self, core: DynamicSchedulingCore):
        base = encode_observation(core, self.max_candidates)
        return encode_graph_observation(
            core, base, self.max_nodes, self.max_edges
        )


class GraphHeuristicSelectionEnv(gym.Env):
    """Use graph context to select among the five heuristic proposals."""

    metadata = {"render_modes": []}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.base = GraphSchedulingEnv(*args, **kwargs)
        candidate_space = self.base.observation_space["candidates"]
        self.action_space = gym.spaces.Discrete(len(POLICY_NAMES))
        spaces = dict(self.base.observation_space.spaces)
        spaces["candidates"] = gym.spaces.Box(
            0.0,
            10.0,
            shape=(len(POLICY_NAMES), candidate_space.shape[1]),
            dtype=np.float32,
        )
        spaces["action_mask"] = gym.spaces.Box(
            0, 1, shape=(len(POLICY_NAMES),), dtype=np.int8
        )
        spaces["candidate_nodes"] = gym.spaces.Box(
            0,
            self.base.max_nodes - 1,
            shape=(len(POLICY_NAMES),),
            dtype=np.int32,
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self._proposals = ()
        self._proposal_evaluations = 0
        self._observation = None
        self.action_counts = [0 for _ in POLICY_NAMES]
        self.base.result_policy = "gnn-heuristic-maskable-ppo"

    @property
    def core(self):
        return self.base.core

    @property
    def current_candidates(self):
        return self._proposals

    def _refresh(self, base_observation):
        candidates = tuple(self.base.current_candidates)
        rows = np.zeros(self.observation_space["candidates"].shape, np.float32)
        nodes = np.zeros(len(POLICY_NAMES), np.int32)
        mask = np.zeros(len(POLICY_NAMES), np.int8)
        proposals = []
        evaluations = 0
        for index, policy in enumerate(POLICY_NAMES):
            proposal, evaluated = choose_candidate(policy, candidates)
            proposals.append(proposal)
            evaluations += evaluated
            if proposal is not None:
                base_index = candidates.index(proposal)
                rows[index] = base_observation["candidates"][base_index]
                nodes[index] = base_observation["candidate_nodes"][base_index]
                mask[index] = 1
        self._proposals = tuple(proposals)
        self._proposal_evaluations = evaluations
        observation = {
            key: value.copy()
            for key, value in base_observation.items()
            if key not in {"candidates", "candidate_nodes", "action_mask"}
        }
        observation.update(
            {
                "candidates": rows,
                "candidate_nodes": nodes,
                "action_mask": mask,
            }
        )
        self._observation = observation
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_counts = [0 for _ in POLICY_NAMES]
        observation, info = self.base.reset(seed=seed, options=options)
        return self._refresh(observation), info

    def action_masks(self):
        if self._observation is None:
            return np.zeros(len(POLICY_NAMES), dtype=bool)
        return self._observation["action_mask"].astype(bool, copy=True)

    def step(self, action):
        return self.step_with_effort(action, self._proposal_evaluations)

    def step_with_effort(
        self,
        action,
        candidate_evaluations,
        decision_wall_seconds=0.0,
    ):
        del candidate_evaluations
        index = int(action)
        if index < 0 or index >= len(POLICY_NAMES) or not self.action_masks()[index]:
            raise ValueError(f"invalid or masked heuristic action: {action}")
        proposal = self._proposals[index]
        if proposal is None:
            raise AssertionError("valid heuristic action must have a proposal")
        self.action_counts[index] += 1
        base_action = tuple(self.base.current_candidates).index(proposal)
        transition = self.base.step_with_effort(
            base_action,
            candidate_evaluations=self._proposal_evaluations,
            decision_wall_seconds=decision_wall_seconds,
        )
        observation, reward, terminated, truncated, info = transition
        info["heuristic_action_counts"] = {
            policy: self.action_counts[i] for i, policy in enumerate(POLICY_NAMES)
        }
        if terminated:
            self._observation = None
            empty = {
                key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.spaces.items()
            }
            return empty, reward, terminated, truncated, info
        return self._refresh(observation), reward, terminated, truncated, info

    @property
    def final_result(self):
        return self.base.final_result
