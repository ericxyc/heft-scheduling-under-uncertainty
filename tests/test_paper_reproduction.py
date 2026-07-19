"""Regression tests against the published HEFT example."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heft_reproduction.cli import main
from heft_reproduction.paper_example import (
    EXPECTED_MAKESPAN,
    EXPECTED_PRIORITY_ORDER,
    EXPECTED_SCHEDULE,
    EXPECTED_UPWARD_RANKS,
    load_paper_example,
)
from heft_reproduction.scheduler import (
    compute_upward_ranks,
    prioritize_tasks,
    schedule_heft,
)


class PaperReproductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = load_paper_example()

    def test_fixture_shape(self) -> None:
        self.assertEqual(self.workflow.tasks, tuple(range(1, 11)))
        self.assertEqual(self.workflow.processors, ("P1", "P2", "P3"))
        self.assertEqual(len(self.workflow.communication_costs), 15)

    def test_upward_ranks_match_table_one(self) -> None:
        ranks = compute_upward_ranks(self.workflow)
        for task, expected in EXPECTED_UPWARD_RANKS.items():
            self.assertLessEqual(abs(ranks[task] - expected), 0.001)

    def test_priority_order_matches_paper(self) -> None:
        self.assertEqual(
            prioritize_tasks(self.workflow),
            EXPECTED_PRIORITY_ORDER,
        )

    def test_schedule_and_makespan_match_figure_four(self) -> None:
        result = schedule_heft(self.workflow)

        self.assertEqual(result.priority_order, EXPECTED_PRIORITY_ORDER)
        self.assertEqual(result.schedule, EXPECTED_SCHEDULE)
        self.assertAlmostEqual(result.makespan, EXPECTED_MAKESPAN)
        self.assertTrue(result.is_valid, result.validation_errors)

    def test_cli_writes_deterministic_json(self) -> None:
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"

            self.assertEqual(main(["--output", str(first)]), 0)
            self.assertEqual(main(["--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(payload["makespan"], EXPECTED_MAKESPAN)
            self.assertEqual(
                payload["priority_order"],
                list(EXPECTED_PRIORITY_ORDER),
            )
            self.assertTrue(payload["is_valid"])


if __name__ == "__main__":
    unittest.main()
