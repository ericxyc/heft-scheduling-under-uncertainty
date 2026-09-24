"""Reproducibility metadata shared by generated experiment reports."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable


DEFAULT_PACKAGES = (
    "gymnasium",
    "numpy",
    "torch",
    "stable-baselines3",
    "sb3-contrib",
    "matplotlib",
)


def _git_value(arguments: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_provenance(
    project_root: Path | None = None,
    packages: Iterable[str] = DEFAULT_PACKAGES,
) -> dict[str, object]:
    """Return commit, platform, interpreter, and dependency versions."""

    root = (project_root or Path.cwd()).resolve()
    dependencies: dict[str, str | None] = {}
    for package in packages:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    status = _git_value(["status", "--porcelain"], root)
    return {
        "git_commit": _git_value(["rev-parse", "HEAD"], root),
        "git_branch": _git_value(["branch", "--show-current"], root),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
