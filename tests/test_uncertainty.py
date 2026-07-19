"""Tests for runtime-uncertainty sampling and static-plan replay."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heft_reproduction.models import ScheduleEntry, Workflow
from heft_reproduction.scheduler import schedule_heft
from heft_reproduction.uncertainty_cli import main
from heft_reproduction.uncertainty import (
    replay_fixed_plan,
    run_uncertainty_experiment,
    sample_duration_multipliers,
    workflow_with_actual_durations,
)


class UncertaintyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = Workflow(
            tasks=(1, 2, 3),
            processors=("P1", "P2"),
            computation_costs={
                1: {"P1": 1.0, "P2": 2.0},
                2: {"P1": 2.0, "P2": 1.0},
                3: {"P1": 3.0, "P2": 1.0},
            },
            communication_costs={(1, 3): 0.5},
        )

    def test_zero_cv_is_exactly_one(self) -> None:
        self.assertEqual(
            sample_duration_multipliers(self.workflow.tasks, 0.0, 8),
            {1: 1.0, 2: 1.0, 3: 1.0},
        )

    def test_replay_propagates_actual_parent_delay_and_transfer(self) -> None:
        plan = {
            1: ScheduleEntry(1, "P1", 0.0, 1.0),
            2: ScheduleEntry(2, "P2", 0.0, 1.0),
            3: ScheduleEntry(3, "P2", 1.5, 2.5),
        }
        actual = workflow_with_actual_durations(
            self.workflow,
            {1: 3.0, 2: 1.0, 3: 1.0},
        )

        replay = replay_fixed_plan(self.workflow, plan, actual)

        self.assertEqual(replay[1].finish, 3.0)
        self.assertEqual(replay[3].start, 3.5)
        self.assertEqual(replay[3].finish, 4.5)

    def test_zero_uncertainty_matches_static_plan(self) -> None:
        plan = schedule_heft(self.workflow)
        experiment = run_uncertainty_experiment(
            self.workflow,
            plan.schedule,
            cvs=(0.0,),
            trials_per_cv=3,
            seed=9,
        )

        self.assertEqual(experiment.summaries[0].static_plan_mean_makespan, plan.makespan)
        self.assertEqual(experiment.summaries[0].static_plan_std_makespan, 0.0)

    def test_seeded_experiment_is_deterministic(self) -> None:
        plan = schedule_heft(self.workflow)
        first = run_uncertainty_experiment(
            self.workflow, plan.schedule, (0.1, 0.3), 4, 123
        )
        second = run_uncertainty_experiment(
            self.workflow, plan.schedule, (0.1, 0.3), 4, 123
        )
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_cli_writes_deterministic_report(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        trace = project_root / (
            "data/raw/wfcommons/montage-chameleon-2mass-005d-001.json"
        )
        config = project_root / "configs/trace_worker_model.json"
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            args = [
                "--input",
                str(trace),
                "--config",
                str(config),
                "--cvs",
                "0,0.1",
                "--trials",
                "2",
                "--seed",
                "4",
            ]

            self.assertEqual(main([*args, "--output", str(first)]), 0)
            self.assertEqual(main([*args, "--output", str(second)]), 0)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["trials"]), 4)
            self.assertEqual(payload["summaries"][0]["cv"], 0.0)


if __name__ == "__main__":
    unittest.main()
