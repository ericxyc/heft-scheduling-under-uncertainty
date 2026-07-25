"""Correctness tests for the Phase 6 graph policy stack."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from heft_reproduction.rl.graph_environment import GraphSchedulingEnv
from heft_reproduction.rl.graph_environment import GraphHeuristicSelectionEnv
from heft_reproduction.rl.graph_policy import GraphCandidateExtractor
from heft_reproduction.rl.graph_training import (
    linear_entropy_coefficient,
    load_gnn_training_config,
)
from tests.test_rl_environment import _factory


class GraphSchedulingTests(unittest.TestCase):
    def _env(self) -> GraphSchedulingEnv:
        return GraphSchedulingEnv(
            scenario_factory=_factory,
            processors=("P1", "P2"),
            max_candidates=32,
            max_nodes=32,
            max_edges=64,
            reward_scale=10.0,
            reward_mode="jct-progress-potential",
            potential_weight=3.0,
        )

    def test_graph_observation_and_mask_are_valid(self) -> None:
        env = self._env()
        observation, _ = env.reset(seed=11)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertGreater(observation["graph_node_mask"].sum(), 0)
        self.assertGreater(env.action_masks().sum(), 0)
        valid_nodes = observation["candidate_nodes"][env.action_masks()]
        self.assertTrue(np.all(observation["graph_node_mask"][valid_nodes]))

    def test_potential_shaping_preserves_jct_policy_ordering(self) -> None:
        env = self._env()
        env.reset(seed=17)
        generator = np.random.default_rng(4)
        terminated = False
        final_info = {}
        while not terminated:
            action = int(generator.choice(np.flatnonzero(env.action_masks())))
            _, _, terminated, _, final_info = env.step(action)

        self.assertAlmostEqual(
            final_info["shaped_raw_return"]
            - final_info["raw_episode_return"],
            final_info["potential_shaping_offset"],
            places=7,
        )
        self.assertAlmostEqual(final_info["reward_identity_error"], 0.0)

    def test_gnn_policy_produces_a_valid_masked_action(self) -> None:
        env = self._env()
        observation, _ = env.reset(seed=19)
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            device="cpu",
            policy_kwargs={
                "features_extractor_class": GraphCandidateExtractor,
                "features_extractor_kwargs": {
                    "hidden_dim": 16,
                    "candidate_dim": 4,
                    "message_passing_steps": 2,
                },
                "net_arch": {"pi": [32], "vf": [32]},
            },
        )
        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        self.assertTrue(env.action_masks()[int(action)])

    def test_graph_heuristic_environment_is_valid(self) -> None:
        env = GraphHeuristicSelectionEnv(
            scenario_factory=_factory,
            processors=("P1", "P2"),
            max_candidates=32,
            max_nodes=32,
            max_edges=64,
            reward_scale=10.0,
            reward_mode="jct-progress-potential",
            potential_weight=3.0,
        )
        observation, _ = env.reset(seed=29)
        self.assertTrue(env.observation_space.contains(observation))
        terminated = False
        while not terminated:
            action = int(np.flatnonzero(env.action_masks())[0])
            _, _, terminated, _, info = env.step(action)
        self.assertTrue(info["result"].is_valid)

    def test_entropy_schedule_decays_to_configured_floor(self) -> None:
        self.assertAlmostEqual(
            linear_entropy_coefficient(0.02, 0.003, 0.0, 0.7),
            0.02,
        )
        self.assertAlmostEqual(
            linear_entropy_coefficient(0.02, 0.003, 0.35, 0.7),
            0.0115,
        )
        self.assertAlmostEqual(
            linear_entropy_coefficient(0.02, 0.003, 0.7, 0.7),
            0.003,
        )
        self.assertAlmostEqual(
            linear_entropy_coefficient(0.02, 0.003, 1.0, 0.7),
            0.003,
        )

    def test_fair_baseline_removes_greedy_prior(self) -> None:
        config = load_gnn_training_config(
            Path("configs/rl/wfcommons_gnn_hybrid_ppo_fair.json")
        )
        self.assertEqual(config.greedy_prior_logit, 0.0)
        self.assertGreater(
            config.entropy_coefficient,
            config.entropy_coefficient_final,
        )

    def test_legacy_gnn_config_gets_constant_entropy_default(self) -> None:
        config = load_gnn_training_config(
            Path("configs/rl/wfcommons_gnn_hybrid_ppo.json")
        )
        self.assertEqual(
            config.entropy_coefficient,
            config.entropy_coefficient_final,
        )
        self.assertEqual(config.entropy_decay_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
