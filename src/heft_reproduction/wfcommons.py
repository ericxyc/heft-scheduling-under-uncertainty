"""Typed parsing and semantic validation for the WfFormat 1.5 subset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class WfFile:
    id: str
    size_in_bytes: int


@dataclass(frozen=True)
class WfTaskSpec:
    id: str
    name: str
    parents: tuple[str, ...]
    children: tuple[str, ...]
    input_files: tuple[str, ...]
    output_files: tuple[str, ...]


@dataclass(frozen=True)
class WfTaskExecution:
    id: str
    runtime_in_seconds: float
    program: str
    avg_cpu: float | None
    core_count: float | None
    memory_in_bytes: int | None
    machines: tuple[str, ...]


@dataclass(frozen=True)
class WfMachine:
    node_name: str
    system: str | None
    architecture: str | None
    core_count: float | None
    speed_in_mhz: float | None
    memory_in_bytes: int | None


@dataclass(frozen=True)
class WfTrace:
    """A validated workflow specification joined with one execution trace."""

    name: str
    description: str | None
    schema_version: str
    runtime_system: str | None
    tasks: Mapping[str, WfTaskSpec]
    executions: Mapping[str, WfTaskExecution]
    files: Mapping[str, WfFile]
    machines: tuple[WfMachine, ...]
    observed_makespan_in_seconds: float
    source_path: Path

    @property
    def edge_count(self) -> int:
        return sum(len(task.children) for task in self.tasks.values())

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(
            sorted(task.id for task in self.tasks.values() if not task.parents)
        )

    @property
    def leaves(self) -> tuple[str, ...]:
        return tuple(
            sorted(task.id for task in self.tasks.values() if not task.children)
        )

    def shared_file_ids(self, parent: str, child: str) -> tuple[str, ...]:
        if parent not in self.tasks or child not in self.tasks:
            raise KeyError(f"unknown dependency {parent}->{child}")
        parent_outputs = set(self.tasks[parent].output_files)
        child_inputs = set(self.tasks[child].input_files)
        return tuple(sorted(parent_outputs & child_inputs))

    def shared_bytes(self, parent: str, child: str) -> int:
        return sum(
            self.files[file_id].size_in_bytes
            for file_id in self.shared_file_ids(parent, child)
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "task_count": len(self.tasks),
            "edge_count": self.edge_count,
            "file_count": len(self.files),
            "machine_count": len(self.machines),
            "root_count": len(self.roots),
            "leaf_count": len(self.leaves),
            "observed_makespan_in_seconds": (
                self.observed_makespan_in_seconds
            ),
            "total_observed_task_runtime_in_seconds": sum(
                execution.runtime_in_seconds
                for execution in self.executions.values()
            ),
        }


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    values = _list(value, path)
    result = tuple(_string(item, f"{path}[]") for item in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{path} contains duplicate IDs")
    return result


def _number(
    value: Any,
    path: str,
    *,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _integer(
    value: Any,
    path: str,
    *,
    optional: bool = False,
) -> int | None:
    number = _number(value, path, optional=optional)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"{path} must be an integer")
    return int(number)


def _unique_by_id(records: list[Any], path: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        if record.id in result:
            raise ValueError(f"{path} contains duplicate ID {record.id}")
        result[record.id] = record
    return result


def _parse_task_spec(value: Any, index: int) -> WfTaskSpec:
    path = f"workflow.specification.tasks[{index}]"
    item = _mapping(value, path)
    return WfTaskSpec(
        id=_string(item.get("id"), f"{path}.id"),
        name=_string(item.get("name"), f"{path}.name"),
        parents=_string_tuple(item.get("parents"), f"{path}.parents"),
        children=_string_tuple(item.get("children"), f"{path}.children"),
        input_files=_string_tuple(
            item.get("inputFiles"), f"{path}.inputFiles"
        ),
        output_files=_string_tuple(
            item.get("outputFiles"), f"{path}.outputFiles"
        ),
    )


def _parse_task_execution(value: Any, index: int) -> WfTaskExecution:
    path = f"workflow.execution.tasks[{index}]"
    item = _mapping(value, path)
    command = _mapping(item.get("command"), f"{path}.command")
    runtime = _number(item.get("runtimeInSeconds"), f"{path}.runtimeInSeconds")
    assert runtime is not None
    if runtime <= 0:
        raise ValueError(f"{path}.runtimeInSeconds must be positive")

    avg_cpu = _number(item.get("avgCPU"), f"{path}.avgCPU", optional=True)
    core_count = _number(
        item.get("coreCount"), f"{path}.coreCount", optional=True
    )
    memory = _integer(
        item.get("memoryInBytes"),
        f"{path}.memoryInBytes",
        optional=True,
    )
    return WfTaskExecution(
        id=_string(item.get("id"), f"{path}.id"),
        runtime_in_seconds=runtime,
        program=_string(command.get("program"), f"{path}.command.program"),
        avg_cpu=avg_cpu,
        core_count=core_count,
        memory_in_bytes=memory,
        machines=_string_tuple(item.get("machines", []), f"{path}.machines"),
    )


def _parse_file(value: Any, index: int) -> WfFile:
    path = f"workflow.specification.files[{index}]"
    item = _mapping(value, path)
    size = _integer(item.get("sizeInBytes"), f"{path}.sizeInBytes")
    assert size is not None
    if size < 0:
        raise ValueError(f"{path}.sizeInBytes must be non-negative")
    return WfFile(
        id=_string(item.get("id"), f"{path}.id"),
        size_in_bytes=size,
    )


def _parse_machine(value: Any, index: int) -> WfMachine:
    path = f"workflow.execution.machines[{index}]"
    item = _mapping(value, path)
    cpu = _mapping(item.get("cpu", {}), f"{path}.cpu")
    return WfMachine(
        node_name=_string(item.get("nodeName"), f"{path}.nodeName"),
        system=item.get("system") if isinstance(item.get("system"), str) else None,
        architecture=(
            item.get("architecture")
            if isinstance(item.get("architecture"), str)
            else None
        ),
        core_count=_number(
            cpu.get("coreCount"), f"{path}.cpu.coreCount", optional=True
        ),
        speed_in_mhz=_number(
            cpu.get("speedInMHz"), f"{path}.cpu.speedInMHz", optional=True
        ),
        memory_in_bytes=_integer(
            item.get("memoryInBytes"),
            f"{path}.memoryInBytes",
            optional=True,
        ),
    )


def _validate_semantics(
    tasks: Mapping[str, WfTaskSpec],
    executions: Mapping[str, WfTaskExecution],
    files: Mapping[str, WfFile],
) -> None:
    task_ids = set(tasks)
    execution_ids = set(executions)
    if task_ids != execution_ids:
        missing = sorted(task_ids - execution_ids)
        unexpected = sorted(execution_ids - task_ids)
        raise ValueError(
            "specification/execution task IDs differ: "
            f"missing execution={missing}, unexpected execution={unexpected}"
        )

    file_ids = set(files)
    for task in tasks.values():
        unknown_parents = sorted(set(task.parents) - task_ids)
        unknown_children = sorted(set(task.children) - task_ids)
        if unknown_parents:
            raise ValueError(
                f"task {task.id} references unknown parents {unknown_parents}"
            )
        if unknown_children:
            raise ValueError(
                f"task {task.id} references unknown children {unknown_children}"
            )

        unknown_files = sorted(
            (set(task.input_files) | set(task.output_files)) - file_ids
        )
        if unknown_files:
            raise ValueError(
                f"task {task.id} references unknown files {unknown_files}"
            )

    for task in tasks.values():
        for parent in task.parents:
            if task.id not in tasks[parent].children:
                raise ValueError(
                    f"inconsistent dependency {parent}->{task.id}: "
                    "missing from parent children"
                )
        for child in task.children:
            if task.id not in tasks[child].parents:
                raise ValueError(
                    f"inconsistent dependency {task.id}->{child}: "
                    "missing from child parents"
                )

    indegree = {task_id: len(task.parents) for task_id, task in tasks.items()}
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        task_id = ready.pop(0)
        visited += 1
        for child in sorted(tasks[task_id].children):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if visited != len(tasks):
        raise ValueError("workflow dependencies must form a DAG")


def load_wfcommons_trace(path: str | Path) -> WfTrace:
    """Load and validate a WfFormat 1.5 workflow execution instance."""

    source_path = Path(path)
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read trace {source_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source_path}: {exc}") from exc

    root = _mapping(raw, "root")
    schema_version = str(root.get("schemaVersion"))
    if schema_version != "1.5":
        raise ValueError(
            f"unsupported WfFormat schema {schema_version}; expected 1.5"
        )

    workflow = _mapping(root.get("workflow"), "workflow")
    specification = _mapping(
        workflow.get("specification"), "workflow.specification"
    )
    execution = _mapping(workflow.get("execution"), "workflow.execution")

    task_records = [
        _parse_task_spec(value, index)
        for index, value in enumerate(
            _list(
                specification.get("tasks"),
                "workflow.specification.tasks",
            )
        )
    ]
    execution_records = [
        _parse_task_execution(value, index)
        for index, value in enumerate(
            _list(execution.get("tasks"), "workflow.execution.tasks")
        )
    ]
    file_records = [
        _parse_file(value, index)
        for index, value in enumerate(
            _list(
                specification.get("files"),
                "workflow.specification.files",
            )
        )
    ]
    machine_records = tuple(
        _parse_machine(value, index)
        for index, value in enumerate(
            _list(execution.get("machines"), "workflow.execution.machines")
        )
    )

    tasks = _unique_by_id(task_records, "workflow.specification.tasks")
    executions = _unique_by_id(
        execution_records, "workflow.execution.tasks"
    )
    files = _unique_by_id(file_records, "workflow.specification.files")
    _validate_semantics(tasks, executions, files)

    makespan = _number(
        execution.get("makespanInSeconds"),
        "workflow.execution.makespanInSeconds",
    )
    assert makespan is not None
    if makespan <= 0:
        raise ValueError(
            "workflow.execution.makespanInSeconds must be positive"
        )

    runtime_system_raw = root.get("runtimeSystem")
    runtime_system = None
    if isinstance(runtime_system_raw, dict):
        value = runtime_system_raw.get("name")
        runtime_system = value if isinstance(value, str) else None

    description = root.get("description")
    return WfTrace(
        name=_string(root.get("name"), "name"),
        description=description if isinstance(description, str) else None,
        schema_version=schema_version,
        runtime_system=runtime_system,
        tasks=tasks,
        executions=executions,
        files=files,
        machines=machine_records,
        observed_makespan_in_seconds=makespan,
        source_path=source_path,
    )
