"""Tests for the Phase 4B corpus, policies, and benchmark sweep."""

from __future__ import annotations

from pathlib import Path
import unittest

from heft_reproduction.benchmark_data import load_benchmark_corpus
from heft_reproduction.dynamic_benchmark import (
    mean_interarrival_for_load,
    run_dynamic_benchmark,
)
from heft_reproduction.dynamic_models import (
    AGING_ROLLING_HEFT,
    SHORTEST_REMAINING_WORK,
)
from heft_reproduction.dynamic_policies import (
    SchedulingCandidate,
    choose_candidate,
)
from heft_reproduction.trace_model import load_trace_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "configs/workflow_benchmark.json"
CONFIG_PATH = PROJECT_ROOT / "configs/trace_worker_model.json"


def _candidate(
    workflow_id: str,
    aging_score: float,
    remaining_work: float,
) -> SchedulingCandidate:
    return SchedulingCandidate(
        ref=(workflow_id, 1),
        processor="P1",
        processor_index=0,
        workflow_arrival=0.0,
        estimated_finish=2.0,
        upward_rank=10.0,
        normalized_upward_rank=0.5,
        aging_score=aging_score,
        workflow_remaining_work=remaining_work,
        static_allowed=True,
        static_absolute_planned_start=0.0,
    )


class Phase4BBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_trace_model_config(CONFIG_PATH)
        cls.corpus = load_benchmark_corpus(MANIFEST_PATH, cls.config)

    def test_manifest_verifies_three_families_and_two_sizes(self) -> None:
        self.assertEqual(len(self.corpus.entries), 6)
        self.assertEqual(self.corpus.sizes, ("medium", "small"))
        self.assertEqual(
            {entry.family for entry in self.corpus.entries},
            {"montage", "epigenomics", "seismology"},
        )
        self.assertEqual(
            [len(template.workflow.tasks) for template in self.corpus.templates],
            [58, 310, 73, 445, 101, 201],
        )

    def test_load_conversion_is_inverse_in_offered_load(self) -> None:
        templates = self.corpus.templates_for_size("small")
        lower_load = mean_interarrival_for_load(templates, 3, 0.5)
        higher_load = mean_interarrival_for_load(templates, 3, 1.0)

        self.assertAlmostEqual(lower_load, 2.0 * higher_load)

    def test_aging_policy_prefers_higher_age_adjusted_score(self) -> None:
        newer = _candidate("W0002", aging_score=0.8, remaining_work=10.0)
        older = _candidate("W0001", aging_score=1.2, remaining_work=20.0)

        selected, evaluated = choose_candidate(
            AGING_ROLLING_HEFT,
            (newer, older),
        )

        self.assertEqual(selected, older)
        self.assertEqual(evaluated, 2)

    def test_shortest_remaining_work_prefers_completion(self) -> None:
        large = _candidate("W0001", aging_score=2.0, remaining_work=20.0)
        small = _candidate("W0002", aging_score=0.5, remaining_work=5.0)

        selected, evaluated = choose_candidate(
            SHORTEST_REMAINING_WORK,
            (large, small),
        )

        self.assertEqual(selected, small)
        self.assertEqual(evaluated, 2)

    def test_sweep_shares_scenario_seed_across_policies(self) -> None:
        benchmark = run_dynamic_benchmark(
            corpus=self.corpus,
            sizes=("small",),
            loads=(0.8,),
            cvs=(0.1,),
            replicate_count=1,
            workflow_count=3,
            base_seed=33,
            policies=(AGING_ROLLING_HEFT, SHORTEST_REMAINING_WORK),
        )

        self.assertTrue(benchmark.is_valid)
        self.assertEqual(len(benchmark.runs), 2)
        self.assertEqual(
            {run.scenario_seed for run in benchmark.runs},
            {33},
        )
        self.assertEqual(
            {run.metrics.task_count for run in benchmark.runs},
            {232},
        )
        self.assertEqual(len(benchmark.aggregates), 2)


if __name__ == "__main__":
    unittest.main()
