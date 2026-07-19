"""Correctness tests for the optional Phase 5 Gymnasium environment."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from heft_reproduction.dynamic_models import WorkflowTemplate
from heft_reproduction.dynamic_models import POLICY_NAMES
from heft_reproduction.dynamic_scenario import build_dynamic_scenario
from heft_reproduction.dynamic_simulator import simulate_dynamic_scenario
from heft_reproduction.models import Workflow


GYM_AVAILABLE = importlib.util.find_spec("gymnasium") is not None


def _template() -> WorkflowTemplate:
    workflow = Workflow(
        tasks=(1, 2, 3, 4),
        processors=("P1", "P2"),
        computation_costs={
            1: {"P1": 1.0, "P2": 2.0},
            2: {"P1": 3.0, "P2": 1.0},
            3: {"P1": 2.0, "P2": 1.5},
            4: {"P1": 1.0, "P2": 2.0},
        },
        communication_costs={(1, 3): 0.5, (2, 3): 0.5, (3, 4): 0.25},
    )
    return WorkflowTemplate(
        name="rl-test",
        workflow=workflow,
        source_task_ids={task: f"T{task}" for task in workflow.tasks},
        programs={task: "test" for task in workflow.tasks},
        edge_data_bytes={
            edge: int(cost * 1000)
            for edge, cost in workflow.communication_costs.items()
        },
        source_filename="synthetic.json",
    )


def _factory(seed: int, cv: float = 0.2):
    return build_dynamic_scenario(
        [_template()],
        workflow_count=3,
        mean_interarrival_time=1.0,
        runtime_cv=cv,
        seed=seed,
    )


@unittest.skipUnless(GYM_AVAILABLE, "Gymnasium is an optional RL dependency")
class DynamicSchedulingEnvTests(unittest.TestCase):
    def _env(self, cv: float = 0.2):
        from heft_reproduction.rl.environment import DynamicSchedulingEnv

        return DynamicSchedulingEnv(
            scenario_factory=lambda seed: _factory(seed, cv=cv),
            processors=("P1", "P2"),
            max_candidates=32,
        )

    def test_reset_is_seeded_and_observation_matches_space(self) -> None:
        env = self._env()
        first, first_info = env.reset(seed=13)
        second, second_info = env.reset(seed=13)

        self.assertTrue(env.observation_space.contains(first))
        self.assertTrue(env.observation_space.contains(second))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first_info["scenario_seed"], second_info["scenario_seed"])
        self.assertGreater(env.action_masks().sum(), 0)

    def test_invalid_masked_action_is_rejected(self) -> None:
        env = self._env()
        env.reset(seed=3)
        invalid = int(np.flatnonzero(~env.action_masks())[0])

        with self.assertRaisesRegex(ValueError, "invalid or masked"):
            env.step(invalid)

    def test_random_masked_episode_is_valid_and_reward_equals_jct(self) -> None:
        env = self._env()
        env.reset(seed=7)
        generator = np.random.default_rng(99)
        terminated = False
        raw_return = 0.0
        final_info = {}
        while not terminated:
            valid = np.flatnonzero(env.action_masks())
            action = int(generator.choice(valid))
            _, reward, terminated, truncated, final_info = env.step(action)
            self.assertFalse(truncated)
            raw_return += reward

        result = final_info["result"]
        self.assertTrue(result.is_valid, result.validation_errors)
        self.assertEqual(result.metrics.task_count, 12)
        self.assertAlmostEqual(raw_return, -final_info["total_jct"], places=9)
        self.assertAlmostEqual(
            final_info["reward_identity_error"],
            0.0,
            places=9,
        )

    def test_observation_does_not_reveal_sampled_actual_duration(self) -> None:
        env = self._env(cv=0.5)
        first, _ = env.reset(seed=1)
        first_actual = [
            instance.actual_workflow.computation_cost(1, "P1")
            for instance in env.core.scenario.instances
        ]
        second, _ = env.reset(seed=2)
        second_actual = [
            instance.actual_workflow.computation_cost(1, "P1")
            for instance in env.core.scenario.instances
        ]

        self.assertNotEqual(first_actual, second_actual)
        np.testing.assert_array_equal(first["candidates"], second["candidates"])

    def test_heuristic_adapters_reproduce_direct_schedule(self) -> None:
        from heft_reproduction.rl.adapters import run_heuristic_episode

        scenario_seed = 23
        for policy in POLICY_NAMES:
            with self.subTest(policy=policy):
                env = self._env()
                adapted = run_heuristic_episode(env, policy, scenario_seed)
                direct = simulate_dynamic_scenario(
                    _factory(scenario_seed),
                    policy,
                )
                self.assertEqual(adapted.result.tasks, direct.tasks)
                self.assertEqual(adapted.result.workflows, direct.workflows)
                self.assertEqual(
                    adapted.result.metrics.mean_jct,
                    direct.metrics.mean_jct,
                )
                self.assertTrue(
                    adapted.result.is_valid,
                    adapted.result.validation_errors,
                )

    def test_hybrid_environment_selects_only_heuristic_proposals(self) -> None:
        from heft_reproduction.rl.adapters import run_random_episode
        from heft_reproduction.rl.hybrid_environment import (
            HeuristicSelectionEnv,
        )

        env = HeuristicSelectionEnv(
            scenario_factory=lambda seed: _factory(seed),
            processors=("P1", "P2"),
            max_candidates=32,
        )
        observation, _ = env.reset(seed=31)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertGreater(env.action_masks().sum(), 0)
        outcome = run_random_episode(env, scenario_seed=31, policy_seed=9)
        self.assertTrue(outcome.result.is_valid)
        self.assertAlmostEqual(
            outcome.raw_return,
            -sum(workflow.jct for workflow in outcome.result.workflows),
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
