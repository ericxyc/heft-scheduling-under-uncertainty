"""Held-out evaluation of a frozen RL scheduler and Phase 4B baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Sequence

from sb3_contrib import MaskablePPO

from ..benchmark_data import BenchmarkCorpus, load_benchmark_corpus
from ..dynamic_benchmark import mean_interarrival_for_load
from ..dynamic_models import POLICY_NAMES, DynamicScenario
from ..dynamic_scenario import build_dynamic_scenario
from ..dynamic_simulator import simulate_dynamic_scenario
from ..trace_model import load_trace_model_config
from .adapters import run_model_episode, run_random_episode
from .environment import DynamicSchedulingEnv


DEFAULT_CONFIG = Path("configs/rl/wfcommons_evaluation.json")
DEFAULT_MANIFEST = Path("configs/workflow_benchmark.json")
DEFAULT_WORKER_CONFIG = Path("configs/trace_worker_model.json")
DEFAULT_MODEL = Path(
    "artifacts/rl/final_models/wfcommons_maskable_ppo.zip"
)
DEFAULT_OUTPUT = Path("results/rl/wfcommons_evaluation.json")
DEFAULT_PLOT = Path("results/rl/wfcommons_evaluation.png")
EVALUATION_POLICIES = ("random-masked", *POLICY_NAMES, "maskable-ppo")
METRICS = (
    "mean_jct",
    "p95_jct",
    "mean_task_queue_wait",
    "average_utilization",
    "scheduler_wall_seconds",
)


@dataclass(frozen=True)
class EvaluationConfig:
    environment_type: str
    base_seed: int
    small_seed_count: int
    medium_seed_count: int
    workflow_count: int
    max_candidates: int
    reward_scale: float
    small_loads: tuple[float, ...]
    small_cvs: tuple[float, ...]
    medium_loads: tuple[float, ...]
    medium_cvs: tuple[float, ...]

    def validate(self) -> None:
        if self.environment_type not in ("candidate", "hybrid"):
            raise ValueError("environment type must be candidate or hybrid")
        if any(
            value <= 0
            for value in (
                self.small_seed_count,
                self.medium_seed_count,
                self.workflow_count,
                self.max_candidates,
            )
        ):
            raise ValueError("evaluation counts must be positive")
        if not isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward scale must be positive and finite")
        for values, allow_zero, name in (
            (self.small_loads, False, "small loads"),
            (self.medium_loads, False, "medium loads"),
            (self.small_cvs, True, "small CVs"),
            (self.medium_cvs, True, "medium CVs"),
        ):
            if not values:
                raise ValueError(f"{name} must be non-empty")
            if any(
                not isfinite(value)
                or value < 0
                or (not allow_zero and value == 0)
                for value in values
            ):
                raise ValueError(f"{name} contain an invalid value")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RL evaluation config must be a JSON object")
    for key in (
        "small_loads",
        "small_cvs",
        "medium_loads",
        "medium_cvs",
    ):
        payload[key] = tuple(payload.get(key, ()))
    expected = set(EvaluationConfig.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(
            f"RL evaluation config fields must be exactly {sorted(expected)}"
        )
    config = EvaluationConfig(**payload)
    config.validate()
    return config


def _fixed_env(
    scenario: DynamicScenario,
    config: EvaluationConfig,
) -> DynamicSchedulingEnv:
    if config.environment_type == "hybrid":
        from .hybrid_environment import HeuristicSelectionEnv

        environment_class = HeuristicSelectionEnv
    else:
        environment_class = DynamicSchedulingEnv
    return environment_class(
        scenario_factory=lambda seed: scenario,
        processors=scenario.processors,
        max_candidates=config.max_candidates,
        reward_scale=config.reward_scale,
    )


def _run_row(
    size: str,
    offered_load: float,
    runtime_cv: float,
    replicate: int,
    scenario_seed: int,
    mean_interarrival: float,
    policy: str,
    result,
    truncated_candidates: int = 0,
    policy_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "size": size,
        "offered_load": offered_load,
        "runtime_cv": runtime_cv,
        "replicate": replicate,
        "scenario_seed": scenario_seed,
        "mean_interarrival_time": mean_interarrival,
        "policy": policy,
        "metrics": result.metrics.to_dict(),
        "is_valid": result.is_valid,
        "validation_errors": list(result.validation_errors),
        "truncated_candidates": truncated_candidates,
        "policy_details": policy_details or {},
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["size"],
            row["offered_load"],
            row["runtime_cv"],
            row["policy"],
        )
        groups.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        statistics: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            values = [float(row["metrics"][metric]) for row in group]
            sample_std = stdev(values) if len(values) > 1 else 0.0
            statistics[metric] = {
                "mean": fmean(values),
                "sample_std": sample_std,
                "ci95_half_width": (
                    1.96 * sample_std / sqrt(len(values))
                ),
            }
        aggregates.append(
            {
                "size": key[0],
                "offered_load": key[1],
                "runtime_cv": key[2],
                "policy": key[3],
                "replicate_count": len(group),
                "statistics": statistics,
                "all_valid": all(row["is_valid"] for row in group),
                "mean_truncated_candidates": fmean(
                    row["truncated_candidates"] for row in group
                ),
            }
        )
    return aggregates


def _comparisons(
    aggregates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    cells = sorted(
        {
            (
                item["size"],
                item["offered_load"],
                item["runtime_cv"],
            )
            for item in aggregates
        }
    )
    results: list[dict[str, Any]] = []
    for size, load, cv in cells:
        cell = [
            item
            for item in aggregates
            if (
                item["size"],
                item["offered_load"],
                item["runtime_cv"],
            )
            == (size, load, cv)
        ]
        learned = next(
            item for item in cell if item["policy"] == "maskable-ppo"
        )
        heuristics = [
            item for item in cell if item["policy"] in POLICY_NAMES
        ]
        best = min(
            heuristics,
            key=lambda item: item["statistics"]["mean_jct"]["mean"],
        )
        learned_jct = learned["statistics"]["mean_jct"]["mean"]
        best_jct = best["statistics"]["mean_jct"]["mean"]
        results.append(
            {
                "size": size,
                "offered_load": load,
                "runtime_cv": cv,
                "learned_mean_jct": learned_jct,
                "best_heuristic": best["policy"],
                "best_heuristic_mean_jct": best_jct,
                "learned_improvement_over_best_heuristic": (
                    1.0 - learned_jct / best_jct
                ),
            }
        )
    return results


def evaluate_wfcommons_model(
    model: MaskablePPO,
    corpus: BenchmarkCorpus,
    config: EvaluationConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    settings = (
        ("small", config.small_loads, config.small_cvs, config.small_seed_count),
        (
            "medium",
            config.medium_loads,
            config.medium_cvs,
            config.medium_seed_count,
        ),
    )
    for size_index, (size, loads, cvs, seed_count) in enumerate(settings):
        templates = corpus.templates_for_size(size)
        for load_index, load in enumerate(loads):
            mean_interarrival = mean_interarrival_for_load(
                templates,
                config.workflow_count,
                load,
            )
            for cv_index, cv in enumerate(cvs):
                for replicate in range(seed_count):
                    scenario_seed = (
                        config.base_seed
                        + size_index * 1_000_000
                        + load_index * 100_000
                        + cv_index * 10_000
                        + replicate
                    )
                    scenario = build_dynamic_scenario(
                        templates=templates,
                        workflow_count=config.workflow_count,
                        mean_interarrival_time=mean_interarrival,
                        runtime_cv=cv,
                        seed=scenario_seed,
                    )
                    env = _fixed_env(scenario, config)
                    random_outcome = run_random_episode(
                        env,
                        scenario_seed,
                        config.base_seed + scenario_seed,
                    )
                    rows.append(
                        _run_row(
                            size,
                            load,
                            cv,
                            replicate,
                            scenario_seed,
                            mean_interarrival,
                            "random-masked",
                            random_outcome.result,
                            random_outcome.truncated_candidates,
                        )
                    )
                    for policy in POLICY_NAMES:
                        result = simulate_dynamic_scenario(scenario, policy)
                        rows.append(
                            _run_row(
                                size,
                                load,
                                cv,
                                replicate,
                                scenario_seed,
                                mean_interarrival,
                                policy,
                                result,
                            )
                        )
                    learned_outcome = run_model_episode(
                        env,
                        model,
                        scenario_seed,
                    )
                    rows.append(
                        _run_row(
                            size,
                            load,
                            cv,
                            replicate,
                            scenario_seed,
                            mean_interarrival,
                            "maskable-ppo",
                            learned_outcome.result,
                            learned_outcome.truncated_candidates,
                            {
                                "heuristic_action_counts": (
                                    learned_outcome.final_info.get(
                                        "heuristic_action_counts",
                                        {},
                                    )
                                )
                            },
                        )
                    )
    aggregates = _aggregate(rows)
    return {
        "rows": rows,
        "aggregates": aggregates,
        "comparisons": _comparisons(aggregates),
        "all_valid": all(row["is_valid"] for row in rows),
    }


def save_evaluation_plot(
    evaluation: dict[str, Any],
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for evaluation plots") from exc
    aggregates = evaluation["aggregates"]
    cells = sorted(
        {
            (
                item["size"],
                item["offered_load"],
                item["runtime_cv"],
            )
            for item in aggregates
        }
    )
    figure, axes = plt.subplots(
        len(cells),
        1,
        figsize=(11, max(4.5, 3.5 * len(cells))),
        squeeze=False,
    )
    labels = {
        "random-masked": "Random",
        "online-greedy-eft": "Greedy",
        "per-workflow-static-heft": "Static",
        "rolling-heft": "Rolling",
        "aging-rolling-heft": "Aging",
        "shortest-remaining-work": "SRW",
        "maskable-ppo": "RL",
    }
    colors = [
        "#9AA0A6",
        "#2364AA",
        "#C44E52",
        "#4C956C",
        "#7B5EA7",
        "#B58B2A",
        "#111111",
    ]
    for axis, cell in zip(axes[:, 0], cells):
        values = [
            next(
                item
                for item in aggregates
                if (
                    item["size"],
                    item["offered_load"],
                    item["runtime_cv"],
                    item["policy"],
                )
                == (*cell, policy)
            )
            for policy in EVALUATION_POLICIES
        ]
        means = [
            item["statistics"]["mean_jct"]["mean"] for item in values
        ]
        errors = [
            item["statistics"]["mean_jct"]["ci95_half_width"]
            for item in values
        ]
        axis.bar(
            range(len(values)),
            means,
            yerr=errors,
            color=colors,
            capsize=4,
        )
        axis.set_xticks(
            range(len(values)),
            [labels[item["policy"]] for item in values],
        )
        axis.set_ylabel("Mean JCT")
        axis.set_title(
            f"{cell[0].title()} | load={cell[1]:.1f} | CV={cell[2]:.1f}"
        )
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen RL scheduler against Phase 4B baselines."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--worker-config",
        type=Path,
        default=DEFAULT_WORKER_CONFIG,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_evaluation_config(args.config)
        worker_config = load_trace_model_config(args.worker_config)
        corpus = load_benchmark_corpus(args.manifest, worker_config)
        model = MaskablePPO.load(args.model, device="cpu")
        evaluation = evaluate_wfcommons_model(model, corpus, config)
        payload = {
            "phase": "5D-held-out-evaluation",
            "model_path": str(args.model),
            "config": {
                **asdict(config),
                "small_loads": list(config.small_loads),
                "small_cvs": list(config.small_cvs),
                "medium_loads": list(config.medium_loads),
                "medium_cvs": list(config.medium_cvs),
            },
            "corpus": corpus.to_dict(),
            "evaluation": evaluation,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_evaluation_plot(evaluation, args.plot)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    for comparison in evaluation["comparisons"]:
        print(
            f"{comparison['size']} load={comparison['offered_load']:.1f} "
            f"CV={comparison['runtime_cv']:.1f}: "
            f"RL JCT={comparison['learned_mean_jct']:.3f} | "
            f"best={comparison['best_heuristic']} "
            f"{comparison['best_heuristic_mean_jct']:.3f} | "
            f"delta={comparison['learned_improvement_over_best_heuristic']:.1%}"
        )
    print(f"All schedules valid: {evaluation['all_valid']}")
    print(f"JSON: {args.output}")
    print(f"Plot: {args.plot}")
    return 0 if evaluation["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
