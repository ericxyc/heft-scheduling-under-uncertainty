"""Build a checksum-pinned, instance-disjoint WfCommons research corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .wfcommons import load_wfcommons_trace


TREE_URL = (
    "https://api.github.com/repos/wfcommons/WfInstances/git/trees/main"
    "?recursive=1"
)
RAW_ROOT = "https://raw.githubusercontent.com/wfcommons/WfInstances/main"
DEFAULT_FAMILIES = (
    "1000genome",
    "cycles",
    "epigenomics",
    "montage",
    "srasearch",
)


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "heft-research-corpus-builder"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"cannot download {url}: {exc}") from exc


def discover_instances(families: Sequence[str]) -> dict[str, list[str]]:
    payload = json.loads(_download(TREE_URL))
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError("GitHub tree response has no file list")
    requested = {value.lower() for value in families}
    result = {family: [] for family in sorted(requested)}
    for item in tree:
        path = item.get("path") if isinstance(item, dict) else None
        if not isinstance(path, str) or not path.endswith(".json"):
            continue
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "pegasus":
            continue
        family = parts[1].lower()
        if family in result:
            result[family].append(path)
    missing = [family for family, paths in result.items() if not paths]
    if missing:
        raise ValueError(f"no WfCommons instances found for: {missing}")
    return {family: sorted(paths) for family, paths in result.items()}


def _split(index: int, count: int) -> str:
    train_end = min(max(1, round(count * 0.70)), count - 2)
    validation_end = min(
        max(train_end + 1, round(count * 0.85)),
        count - 1,
    )
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def build_corpus(
    families: Sequence[str],
    limit_per_family: int,
    max_tasks_per_instance: int,
    data_dir: Path,
    output_manifest: Path,
) -> dict[str, object]:
    if limit_per_family < 3:
        raise ValueError("limit per family must be at least 3 for three splits")
    if max_tasks_per_instance <= 150:
        raise ValueError(
            "max tasks per instance must exceed the small/medium boundary"
        )
    discovered = discover_instances(families)
    project_root = output_manifest.resolve().parent.parent
    entries: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for family, available in discovered.items():
        selected: list[dict[str, object]] = []
        for repository_path in available:
            if len(selected) >= limit_per_family:
                break
            filename = Path(repository_path).name
            local_path = (data_dir / family / filename).resolve()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            source_url = f"{RAW_ROOT}/{repository_path}"
            content = _download(source_url)
            digest = hashlib.sha256(content).hexdigest()
            if local_path.exists() and local_path.read_bytes() != content:
                raise ValueError(f"existing corpus file differs: {local_path}")
            temporary_name: str | None = None
            trace_path = local_path
            if not local_path.exists():
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=local_path.parent,
                    prefix=f".{local_path.stem}-",
                    suffix=".json.part",
                    delete=False,
                ) as temporary:
                    temporary.write(content)
                    temporary_name = temporary.name
                trace_path = Path(temporary_name)
            try:
                trace = load_wfcommons_trace(trace_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                rejected.append(
                    {
                        "family": family,
                        "source_url": source_url,
                        "reason": str(exc),
                    }
                )
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
                continue
            task_count = len(trace.tasks)
            if task_count > max_tasks_per_instance:
                rejected.append(
                    {
                        "family": family,
                        "source_url": source_url,
                        "reason": (
                            f"task count {task_count} exceeds configured limit "
                            f"{max_tasks_per_instance}"
                        ),
                    }
                )
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
                continue
            if temporary_name is not None:
                os.replace(temporary_name, local_path)
            selected.append(
                {
                    "name": f"{family}-{Path(filename).stem}",
                    "family": family,
                    "size": "small" if task_count <= 150 else "medium",
                    "path": str(local_path.relative_to(project_root)).replace(
                        "\\", "/"
                    ),
                    "source_url": source_url,
                    "sha256": digest,
                    "task_count": task_count,
                    "edge_count": trace.edge_count,
                    "file_count": len(trace.files),
                }
            )
        if len(selected) < 3:
            raise ValueError(
                f"family {family} has fewer than three compatible instances"
            )
        for index, entry in enumerate(selected):
            entry["split"] = _split(index, len(selected))
            entries.append(entry)
    manifest = {
        "benchmark_name": "wfcommons-instance-disjoint-research-v1",
        "description": (
            "Checksum-pinned WfCommons corpus split by DAG instance into "
            "train, validation, and test partitions."
        ),
        "max_tasks_per_instance": max_tasks_per_instance,
        "entries": entries,
        "rejected_instances": rejected,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and pin an instance-disjoint WfCommons corpus"
    )
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="comma-separated Pegasus application directories",
    )
    parser.add_argument("--limit-per-family", type=int, default=20)
    parser.add_argument("--max-tasks-per-instance", type=int, default=300)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/wfcommons-expanded"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/workflow_research_corpus.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = tuple(
        value.strip().lower()
        for value in args.families.split(",")
        if value.strip()
    )
    try:
        manifest = build_corpus(
            families=families,
            limit_per_family=args.limit_per_family,
            max_tasks_per_instance=args.max_tasks_per_instance,
            data_dir=args.data_dir,
            output_manifest=args.output,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 2
    counts: dict[str, int] = {}
    for entry in manifest["entries"]:
        split = str(entry["split"])
        counts[split] = counts.get(split, 0) + 1
    print(f"Instances: {len(manifest['entries'])} | splits: {counts}")
    print(f"Manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
