"""Tests for Phase 4 dynamic multi-workflow scheduling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heft_reproduction.dynamic_cli import main
from heft_reproduction.dynamic_models import POLICY_NAMES, STATIC_HEFT, WorkflowTemplate
from heft_reproduction.dynamic_scenario import build_dynamic_scenario
from heft_reproduction.dynamic_simulator import simulate_dynamic_scenario
from heft_reproduction.models import Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = PROJECT_ROOT / (
    "data/raw/wfcommons/montage-chameleon-2mass-005d-001.json"
)
CONFIG_PATH = PROJECT_ROOT / "configs/trace_worker_model.json"


def _template(
    workflow: Workflow,
    name: str = "synthetic",
    edge_bytes: dict[tuple[int, int], int] | None = None,
) -> WorkflowTemplate:
    return WorkflowTemplate(
        name=name,
        workflow=workflow,
        source_task_ids={
            task: f"source-{task}" for task in workflow.tasks
        },
        programs={task: "test-program" for task in workflow.tasks},
        edge_data_bytes=edge_bytes or {
            edge: 0 for edge in workflow.communication_costs
        },
        source_filename="synthetic.json",
    )


class DynamicSchedulingTests(unittest.TestCase):
    def test_seeded_scenario_is_deterministic(self) -> None:
        workflow = Workflow(
            tasks=(1,),
            processors=("P1",),
            computation_costs={1: {"P1": 2.0}},
            communication_costs={},
        )
        first = build_dynamic_scenario(
            [_template(workflow)],
            workflow_count=4,
            mean_interarrival_time=3.0,
            runtime_cv=0.3,
            seed=17,
        )
        second = build_dynamic_scenario(
            [_template(workflow)],
            workflow_count=4,
            mean_interarrival_time=3.0,
            runtime_cv=0.3,
            seed=17,
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        for first_instance, second_instance in zip(
            first.instances,
            second.instances,
        ):
            self.assertEqual(
                first_instance.actual_workflow.computation_cost(1, "P1"),
                second_instance.actual_workflow.computation_cost(1, "P1"),
            )

    def test_future_workflow_cannot_start_before_arrival(self) -> None:
        workflow = Workflow(
            tasks=(1,),
            processors=("P1",),
            computation_costs={1: {"P1": 1.0}},
            communication_costs={},
        )
        scenario = build_dynamic_scenario(
            [_template(workflow)],
            workflow_count=2,
            mean_interarrival_time=0.0,
            runtime_cv=0.0,
            seed=1,
        )
        first, second = scenario.instances
        delayed_type = type(second)
        delayed_second = delayed_type(
            id=second.id,
            template=second.template,
            arrival_time=10.0,
            actual_workflow=second.actual_workflow,
            runtime_seed=second.runtime_seed,
        )
        scenario_type = type(scenario)
        delayed = scenario_type(
            instances=(first, delayed_second),
            runtime_cv=0.0,
            mean_interarrival_time=10.0,
            master_seed=1,
        )

        result = simulate_dynamic_scenario(delayed, POLICY_NAMES[0])

        second_task = next(
            task for task in result.tasks if task.workflow_id == second.id
        )
        self.assertGreaterEqual(second_task.start, 10.0)
        self.assertTrue(result.is_valid, result.validation_errors)

    def test_static_heft_respects_cross_worker_transfer(self) -> None:
        workflow = Workflow(
            tasks=(1, 2),
            processors=("P1", "P2"),
            computation_costs={
                1: {"P1": 1.0, "P2": 10.0},
                2: {"P1": 10.0, "P2": 1.0},
            },
            communication_costs={(1, 2): 5.0},
        )
        scenario = build_dynamic_scenario(
            [_template(workflow, edge_bytes={(1, 2): 500})],
            workflow_count=1,
            mean_interarrival_time=0.0,
            runtime_cv=0.0,
            seed=2,
        )

        result = simulate_dynamic_scenario(scenario, STATIC_HEFT)
        tasks = {task.task: task for task in result.tasks}

        self.assertEqual(tasks[1].processor, "P1")
        self.assertEqual(tasks[1].finish, 1.0)
        self.assertEqual(tasks[2].processor, "P2")
        self.assertEqual(tasks[2].start, 6.0)
        self.assertEqual(result.metrics.cross_worker_data_bytes, 500)
        self.assertTrue(result.is_valid, result.validation_errors)

    def test_all_policies_complete_the_same_dynamic_scenario(self) -> None:
        workflow = Workflow(
            tasks=(1, 2, 3),
            processors=("P1", "P2"),
            computation_costs={
                1: {"P1": 1.0, "P2": 2.0},
                2: {"P1": 2.0, "P2": 1.0},
                3: {"P1": 1.0, "P2": 2.0},
            },
            communication_costs={(1, 3): 0.25, (2, 3): 0.25},
        )
        scenario = build_dynamic_scenario(
            [_template(workflow)],
            workflow_count=3,
            mean_interarrival_time=1.0,
            runtime_cv=0.2,
            seed=8,
        )

        results = [
            simulate_dynamic_scenario(scenario, policy)
            for policy in POLICY_NAMES
        ]

        for result in results:
            self.assertTrue(result.is_valid, result.validation_errors)
            self.assertEqual(len(result.tasks), 9)
            self.assertEqual(result.metrics.committed_decisions, 9)
            self.assertEqual(len(result.workflows), 3)

    def test_cli_core_results_are_reproducible(self) -> None:
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            args = [
                "--input",
                str(TRACE_PATH),
                "--config",
                str(CONFIG_PATH),
                "--workflows",
                "2",
                "--mean-interarrival",
                "5",
                "--cv",
                "0.1",
                "--seed",
                "11",
            ]

            self.assertEqual(main([*args, "--output", str(first)]), 0)
            self.assertEqual(main([*args, "--output", str(second)]), 0)
            first_payload = json.loads(first.read_text(encoding="utf-8"))
            second_payload = json.loads(second.read_text(encoding="utf-8"))
            for payload in (first_payload, second_payload):
                for result in payload["policy_results"]:
                    result["metrics"].pop("scheduler_wall_seconds")

            self.assertEqual(first_payload, second_payload)
            self.assertEqual(
                len(first_payload["policy_results"]),
                len(POLICY_NAMES),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "Matplotlib is optional",
    )
    def test_cli_can_create_comparison_plot(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "dynamic.json"
            plot = Path(directory) / "dynamic.png"
            exit_code = main(
                [
                    "--input",
                    str(TRACE_PATH),
                    "--config",
                    str(CONFIG_PATH),
                    "--workflows",
                    "2",
                    "--mean-interarrival",
                    "5",
                    "--cv",
                    "0.1",
                    "--seed",
                    "11",
                    "--output",
                    str(output),
                    "--plot",
                    str(plot),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertGreater(plot.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
