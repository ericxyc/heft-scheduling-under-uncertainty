"""Lightweight tests for Phase 5 training configuration and toy data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


RL_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("gymnasium", "torch", "stable_baselines3", "sb3_contrib")
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(RL_AVAILABLE, "RL training dependencies are optional")
class RLTrainingTests(unittest.TestCase):
    def test_training_config_and_toy_scenario(self) -> None:
        from heft_reproduction.rl.toy import TOY_PROCESSORS, build_toy_scenario
        from heft_reproduction.rl.training import load_training_config

        config = load_training_config(
            PROJECT_ROOT / "configs/rl/toy_maskable_ppo.json"
        )
        scenario = build_toy_scenario(
            seed=config.seed,
            workflow_count=config.workflow_count,
            mean_interarrival_time=config.mean_interarrival_time,
            runtime_cv=config.runtime_cv,
        )

        self.assertEqual(scenario.processors, TOY_PROCESSORS)
        self.assertEqual(len(scenario.instances), config.workflow_count)
        self.assertEqual(scenario.task_count, 6 * config.workflow_count)
        self.assertGreater(config.total_timesteps, 0)

    def test_wfcommons_training_and_evaluation_configs(self) -> None:
        from heft_reproduction.rl.evaluation import load_evaluation_config
        from heft_reproduction.rl.wfcommons_training import (
            load_wf_training_config,
        )

        training = load_wf_training_config(
            PROJECT_ROOT / "configs/rl/wfcommons_maskable_ppo.json"
        )
        evaluation = load_evaluation_config(
            PROJECT_ROOT / "configs/rl/wfcommons_evaluation.json"
        )

        self.assertEqual(training.workflow_count, evaluation.workflow_count)
        self.assertEqual(training.max_candidates, evaluation.max_candidates)
        self.assertGreater(len(training.training_loads), 1)
        self.assertGreater(evaluation.small_seed_count, 1)


if __name__ == "__main__":
    unittest.main()
