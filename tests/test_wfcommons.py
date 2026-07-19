"""Focused tests for WfFormat parsing and semantic validation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from heft_reproduction.wfcommons import load_wfcommons_trace


def minimal_trace_data() -> dict[str, object]:
    return {
        "name": "tiny",
        "description": "two-task workflow",
        "createdAt": "2026-07-19T00:00:00+08:00",
        "schemaVersion": "1.5",
        "author": {},
        "runtimeSystem": {"name": "test"},
        "workflow": {
            "specification": {
                "tasks": [
                    {
                        "id": "A",
                        "name": "A",
                        "parents": [],
                        "children": ["B"],
                        "inputFiles": [],
                        "outputFiles": ["shared.dat"],
                    },
                    {
                        "id": "B",
                        "name": "B",
                        "parents": ["A"],
                        "children": [],
                        "inputFiles": ["shared.dat"],
                        "outputFiles": [],
                    },
                ],
                "files": [
                    {"id": "shared.dat", "sizeInBytes": 100_000_000}
                ],
            },
            "execution": {
                "makespanInSeconds": 5.0,
                "executedAt": "2026-07-19T00:00:00+08:00",
                "tasks": [
                    {
                        "id": "A",
                        "runtimeInSeconds": 2.0,
                        "command": {"program": "compute"},
                        "avgCPU": 100.0,
                        "machines": ["node"],
                    },
                    {
                        "id": "B",
                        "runtimeInSeconds": 3.0,
                        "command": {"program": "io"},
                        "avgCPU": 0.0,
                        "machines": ["node"],
                    },
                ],
                "machines": [
                    {
                        "nodeName": "node",
                        "system": "linux",
                        "architecture": "x86_64",
                        "cpu": {
                            "coreCount": 4,
                            "speedInMHz": 2500,
                        },
                        "memoryInBytes": 8_000_000_000,
                    }
                ],
            },
        },
    }


def write_trace(directory: str, data: dict[str, object]) -> Path:
    path = Path(directory) / "trace.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class WfCommonsParserTests(unittest.TestCase):
    def test_parses_and_joins_valid_trace(self) -> None:
        with TemporaryDirectory() as directory:
            trace = load_wfcommons_trace(
                write_trace(directory, minimal_trace_data())
            )

        self.assertEqual(len(trace.tasks), 2)
        self.assertEqual(trace.edge_count, 1)
        self.assertEqual(trace.roots, ("A",))
        self.assertEqual(trace.leaves, ("B",))
        self.assertEqual(trace.shared_file_ids("A", "B"), ("shared.dat",))
        self.assertEqual(trace.shared_bytes("A", "B"), 100_000_000)
        self.assertEqual(trace.executions["A"].program, "compute")

    def test_rejects_unknown_parent(self) -> None:
        data = deepcopy(minimal_trace_data())
        tasks = data["workflow"]["specification"]["tasks"]  # type: ignore[index]
        tasks[1]["parents"] = ["MISSING"]

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown parents"):
                load_wfcommons_trace(write_trace(directory, data))

    def test_rejects_mismatched_execution_ids(self) -> None:
        data = deepcopy(minimal_trace_data())
        executions = data["workflow"]["execution"]["tasks"]  # type: ignore[index]
        executions.pop()

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "specification/execution task IDs differ",
            ):
                load_wfcommons_trace(write_trace(directory, data))

    def test_rejects_cycles(self) -> None:
        data = deepcopy(minimal_trace_data())
        tasks = data["workflow"]["specification"]["tasks"]  # type: ignore[index]
        tasks[0]["parents"] = ["B"]
        tasks[1]["children"] = ["A"]

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "DAG"):
                load_wfcommons_trace(write_trace(directory, data))


if __name__ == "__main__":
    unittest.main()
