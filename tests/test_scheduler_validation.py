"""Behavioral tests for insertion and independent validation."""

from __future__ import annotations

import unittest

from heft_reproduction.models import ScheduleEntry, Workflow
from heft_reproduction.paper_example import (
    EXPECTED_SCHEDULE,
    load_paper_example,
)
from heft_reproduction.scheduler import (
    earliest_start_on_processor,
    validate_schedule,
)


class SchedulerBehaviorTests(unittest.TestCase):
    def test_task_is_inserted_into_internal_idle_gap(self) -> None:
        workflow = Workflow(
            tasks=(1, 2, 3),
            processors=("P1", "P2"),
            computation_costs={
                1: {"P1": 2.0, "P2": 3.0},
                2: {"P1": 2.0, "P2": 3.0},
                3: {"P1": 2.0, "P2": 3.0},
            },
            communication_costs={},
        )
        partial_schedule = {
            1: ScheduleEntry(1, "P1", 0.0, 2.0),
            2: ScheduleEntry(2, "P1", 5.0, 7.0),
        }

        start = earliest_start_on_processor(
            workflow,
            task=3,
            processor="P1",
            schedule=partial_schedule,
        )

        self.assertEqual(start, 2.0)

    def test_validator_reports_processor_overlap(self) -> None:
        workflow = load_paper_example()
        invalid_schedule = dict(EXPECTED_SCHEDULE)
        invalid_schedule[3] = ScheduleEntry(3, "P3", 8.0, 27.0)

        errors = validate_schedule(workflow, invalid_schedule)

        self.assertTrue(
            any("processor-overlap" in error for error in errors),
            errors,
        )

    def test_workflow_rejects_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "DAG"):
            Workflow(
                tasks=(1, 2),
                processors=("P1",),
                computation_costs={
                    1: {"P1": 1.0},
                    2: {"P1": 1.0},
                },
                communication_costs={(1, 2): 0.0, (2, 1): 0.0},
            )


if __name__ == "__main__":
    unittest.main()
