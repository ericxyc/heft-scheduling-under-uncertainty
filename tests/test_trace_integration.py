"""End-to-end tests for trace modeling, scheduling, reporting, and plotting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heft_reproduction.scheduler import schedule_heft
from heft_reproduction.trace_cli import main
from heft_reproduction.trace_metrics import calculate_trace_metrics
from heft_reproduction.trace_model import (
    WorkerProfile,
    build_modeled_workflow,
    load_trace_model_config,
    modeled_runtime,
)
from heft_reproduction.wfcommons import load_wfcommons_trace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = PROJECT_ROOT / (
    "data/raw/wfcommons/montage-chameleon-2mass-005d-001.json"
)
CONFIG_PATH = PROJECT_ROOT / "configs/trace_worker_model.json"
EXPECTED_SHA256 = (
    "5795e0ab9e13bb7d50d046796bcbc8ec0a884eba0512a95222bbd557bc6d0b65"
)


class TraceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trace = load_wfcommons_trace(TRACE_PATH)
        cls.config = load_trace_model_config(CONFIG_PATH)
        cls.modeled = build_modeled_workflow(cls.trace, cls.config)
        cls.result = schedule_heft(cls.modeled.workflow)
        cls.metrics = calculate_trace_metrics(cls.modeled, cls.result)

    def test_vendored_trace_checksum_and_shape(self) -> None:
        digest = hashlib.sha256(TRACE_PATH.read_bytes()).hexdigest()

        self.assertEqual(digest, EXPECTED_SHA256)
        self.assertEqual(len(self.trace.tasks), 58)
        self.assertEqual(self.trace.edge_count, 114)
        self.assertEqual(len(self.trace.files), 111)
        self.assertEqual(len(self.trace.machines), 1)
        self.assertEqual(self.trace.observed_makespan_in_seconds, 1060.0)

    def test_worker_formula_has_expected_affinities(self) -> None:
        compute_fast = WorkerProfile("compute", 1.6, 0.8)
        io_fast = WorkerProfile("io", 0.8, 1.6)

        self.assertLess(
            modeled_runtime(10.0, 1.0, compute_fast),
            modeled_runtime(10.0, 1.0, io_fast),
        )
        self.assertLess(
            modeled_runtime(10.0, 0.0, io_fast),
            modeled_runtime(10.0, 0.0, compute_fast),
        )
        balanced = next(
            worker for worker in self.config.workers if worker.id == "Balanced"
        )
        self.assertEqual(modeled_runtime(10.0, 0.37, balanced), 10.0)

    def test_file_bytes_become_communication_seconds(self) -> None:
        for edge, data_bytes in self.modeled.edge_data_bytes.items():
            expected = (
                data_bytes
                / self.config.network_bandwidth_bytes_per_second
            )
            self.assertAlmostEqual(
                self.modeled.workflow.communication_cost(*edge),
                expected,
            )

    def test_trace_schedule_is_complete_and_valid(self) -> None:
        self.assertEqual(len(self.result.schedule), 58)
        self.assertTrue(self.result.is_valid, self.result.validation_errors)
        self.assertAlmostEqual(self.result.makespan, 68.952823811875)
        self.assertGreater(self.metrics.average_utilization, 0.9)
        self.assertEqual(self.metrics.cross_worker_edge_count, 47)
        self.assertEqual(
            self.metrics.cross_worker_data_bytes,
            299_311_627,
        )

    def test_cli_writes_deterministic_source_aware_json(self) -> None:
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            args = [
                "--input",
                str(TRACE_PATH),
                "--config",
                str(CONFIG_PATH),
            ]
            self.assertEqual(main([*args, "--output", str(first)]), 0)
            self.assertEqual(main([*args, "--output", str(second)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["schedule"]), 58)
            self.assertEqual(payload["source"]["sha256"], EXPECTED_SHA256)
            self.assertIn("source_id", payload["schedule"][0])
            self.assertTrue(payload["is_valid"])

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "Matplotlib is optional",
    )
    def test_cli_can_create_gantt_chart(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            plot = Path(directory) / "result.png"
            exit_code = main(
                [
                    "--input",
                    str(TRACE_PATH),
                    "--config",
                    str(CONFIG_PATH),
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
