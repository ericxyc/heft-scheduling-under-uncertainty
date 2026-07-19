"""Load and verify the multi-family Phase 4B workflow corpus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .dynamic_models import WorkflowTemplate
from .dynamic_scenario import template_from_modeled
from .trace_model import (
    TraceModelConfig,
    build_modeled_workflow,
)
from .wfcommons import load_wfcommons_trace


@dataclass(frozen=True)
class BenchmarkEntry:
    """One checksum-pinned WfCommons trace in the benchmark manifest."""

    name: str
    family: str
    size: str
    split: str
    path: Path
    source_url: str
    sha256: str
    task_count: int
    edge_count: int
    file_count: int

    def to_dict(self, project_root: Path) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "size": self.size,
            "split": self.split,
            "path": str(self.path.relative_to(project_root)),
            "source_url": self.source_url,
            "sha256": self.sha256,
            "task_count": self.task_count,
            "edge_count": self.edge_count,
            "file_count": self.file_count,
        }


@dataclass(frozen=True)
class BenchmarkCorpus:
    """Verified manifest metadata and trace-derived workflow templates."""

    name: str
    description: str
    manifest_path: Path
    project_root: Path
    entries: tuple[BenchmarkEntry, ...]
    templates: tuple[WorkflowTemplate, ...]

    def templates_for_size(self, size: str) -> tuple[WorkflowTemplate, ...]:
        selected_names = {
            entry.name for entry in self.entries if entry.size == size
        }
        result = tuple(
            template
            for template in self.templates
            if template.name in selected_names
        )
        if not result:
            raise ValueError(f"benchmark contains no templates for size {size}")
        return result

    @property
    def sizes(self) -> tuple[str, ...]:
        return tuple(sorted({entry.size for entry in self.entries}))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "manifest": str(self.manifest_path.relative_to(self.project_root)),
            "entries": [
                entry.to_dict(self.project_root) for entry in self.entries
            ],
        }


def _required_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _required_positive_int(
    raw: dict[str, Any],
    key: str,
    context: str,
) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}.{key} must be a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_corpus(
    manifest_path: str | Path,
    config: TraceModelConfig,
) -> BenchmarkCorpus:
    """Verify manifest files and convert every trace into a template."""

    source_path = Path(manifest_path).resolve()
    project_root = source_path.parent.parent
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read benchmark manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark manifest JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("benchmark manifest root must be an object")

    name = _required_string(raw, "benchmark_name", "manifest")
    description = _required_string(raw, "description", "manifest")
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise ValueError("manifest.entries must be a non-empty array")

    entries: list[BenchmarkEntry] = []
    templates: list[WorkflowTemplate] = []
    for index, value in enumerate(entries_raw):
        context = f"entries[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{context} must be an object")
        relative_path = Path(_required_string(value, "path", context))
        local_path = (project_root / relative_path).resolve()
        try:
            local_path.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(
                f"{context}.path must stay inside the project"
            ) from exc
        entry = BenchmarkEntry(
            name=_required_string(value, "name", context),
            family=_required_string(value, "family", context),
            size=_required_string(value, "size", context),
            split=_required_string(value, "split", context),
            path=local_path,
            source_url=_required_string(value, "source_url", context),
            sha256=_required_string(value, "sha256", context),
            task_count=_required_positive_int(value, "task_count", context),
            edge_count=_required_positive_int(value, "edge_count", context),
            file_count=_required_positive_int(value, "file_count", context),
        )
        if not local_path.is_file():
            raise ValueError(f"benchmark trace does not exist: {local_path}")
        actual_sha256 = _sha256(local_path)
        if actual_sha256 != entry.sha256:
            raise ValueError(
                f"checksum mismatch for {entry.name}: "
                f"{actual_sha256} != {entry.sha256}"
            )

        trace = load_wfcommons_trace(local_path)
        if len(trace.tasks) != entry.task_count:
            raise ValueError(f"task count mismatch for {entry.name}")
        if trace.edge_count != entry.edge_count:
            raise ValueError(f"edge count mismatch for {entry.name}")
        if len(trace.files) != entry.file_count:
            raise ValueError(f"file count mismatch for {entry.name}")
        modeled = build_modeled_workflow(trace, config)
        templates.append(template_from_modeled(modeled, name=entry.name))
        entries.append(entry)

    if len({entry.name for entry in entries}) != len(entries):
        raise ValueError("benchmark entry names must be unique")
    families_by_size: dict[str, set[str]] = {}
    for entry in entries:
        families_by_size.setdefault(entry.size, set()).add(entry.family)
    if any(len(families) < 2 for families in families_by_size.values()):
        raise ValueError("each benchmark size must cover multiple families")

    return BenchmarkCorpus(
        name=name,
        description=description,
        manifest_path=source_path,
        project_root=project_root,
        entries=tuple(entries),
        templates=tuple(templates),
    )
