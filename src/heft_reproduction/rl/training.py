"""Train and validate the first masked PPO scheduling policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from ..dynamic_models import ONLINE_GREEDY, SHORTEST_REMAINING_WORK
from .adapters import (
    EpisodeOutcome,
    run_heuristic_episode,
    run_model_episode,
    run_random_episode,
)
from .environment import DynamicSchedulingEnv
from .toy import TOY_PROCESSORS, build_toy_scenario


DEFAULT_CONFIG = Path("configs/rl/toy_maskable_ppo.json")
DEFAULT_MODEL = Path(
    "artifacts/rl/final_models/toy_maskable_ppo.zip"
)
DEFAULT_OUTPUT = Path("results/rl/toy_training.json")
DEFAULT_PLOT = Path("results/rl/toy_learning_curve.png")


@dataclass(frozen=True)
class ToyTrainingConfig:
    seed: int
    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    entropy_coefficient: float
    max_candidates: int
    reward_scale: float
    workflow_count: int
    mean_interarrival_time: float
    runtime_cv: float
    evaluation_seed_start: int
    evaluation_seed_count: int
    minimum_random_improvement: float

    def validate(self) -> None:
        integer_positive = (
            self.total_timesteps,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.max_candidates,
            self.workflow_count,
            self.evaluation_seed_count,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("positive training counts must be greater than zero")
        finite_values = (
            self.learning_rate,
            self.gamma,
            self.gae_lambda,
            self.entropy_coefficient,
            self.reward_scale,
            self.mean_interarrival_time,
            self.runtime_cv,
            self.minimum_random_improvement,
        )
        if any(not isfinite(value) for value in finite_values):
            raise ValueError("training parameters must be finite")
        if self.learning_rate <= 0 or self.reward_scale <= 0:
            raise ValueError("learning rate and reward scale must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and GAE lambda must be in [0, 1]")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy coefficient must be non-negative")
        if self.mean_interarrival_time < 0 or self.runtime_cv < 0:
            raise ValueError("arrival mean and runtime CV must be non-negative")
        if not 0 <= self.minimum_random_improvement < 1:
            raise ValueError("minimum improvement must be in [0, 1)")


def load_training_config(path: Path) -> ToyTrainingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RL training config must be a JSON object")
    expected = set(ToyTrainingConfig.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(
            f"RL training config fields must be exactly {sorted(expected)}"
        )
    config = ToyTrainingConfig(**payload)
    config.validate()
    return config


def make_toy_env(config: ToyTrainingConfig) -> DynamicSchedulingEnv:
    return DynamicSchedulingEnv(
        scenario_factory=lambda seed: build_toy_scenario(
            seed=seed,
            workflow_count=config.workflow_count,
            mean_interarrival_time=config.mean_interarrival_time,
            runtime_cv=config.runtime_cv,
        ),
        processors=TOY_PROCESSORS,
        max_candidates=config.max_candidates,
        reward_scale=config.reward_scale,
    )


class EpisodeHistoryCallback(BaseCallback):
    """Collect terminal raw JCT objectives without changing training."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.raw_returns: list[float] = []

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", ()),
            self.locals.get("infos", ()),
        ):
            if done and "raw_episode_return" in info:
                self.raw_returns.append(float(info["raw_episode_return"]))
        return True


def _evaluation_row(
    seed: int,
    random_outcome: EpisodeOutcome,
    learned_outcome: EpisodeOutcome,
    greedy_outcome: EpisodeOutcome,
    srw_outcome: EpisodeOutcome,
) -> dict[str, Any]:
    return {
        "scenario_seed": seed,
        "random_mean_jct": random_outcome.result.metrics.mean_jct,
        "learned_mean_jct": learned_outcome.result.metrics.mean_jct,
        "greedy_mean_jct": greedy_outcome.result.metrics.mean_jct,
        "srw_mean_jct": srw_outcome.result.metrics.mean_jct,
        "learned_inference_seconds": (
            learned_outcome.result.metrics.scheduler_wall_seconds
        ),
        "learned_valid": learned_outcome.result.is_valid,
        "reward_identity_error": (
            learned_outcome.raw_return
            + sum(
                workflow.jct
                for workflow in learned_outcome.result.workflows
            )
        ),
    }


def evaluate_toy_model(
    model: MaskablePPO,
    config: ToyTrainingConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for offset in range(config.evaluation_seed_count):
        seed = config.evaluation_seed_start + offset
        env = make_toy_env(config)
        rows.append(
            _evaluation_row(
                seed,
                run_random_episode(env, seed, config.seed + seed),
                run_model_episode(env, model, seed),
                run_heuristic_episode(env, ONLINE_GREEDY, seed),
                run_heuristic_episode(
                    env,
                    SHORTEST_REMAINING_WORK,
                    seed,
                ),
            )
        )

    random_mean = fmean(row["random_mean_jct"] for row in rows)
    learned_mean = fmean(row["learned_mean_jct"] for row in rows)
    greedy_mean = fmean(row["greedy_mean_jct"] for row in rows)
    srw_mean = fmean(row["srw_mean_jct"] for row in rows)
    improvement = 1.0 - learned_mean / random_mean
    gate_passed = (
        improvement >= config.minimum_random_improvement
        and all(row["learned_valid"] for row in rows)
        and all(
            abs(row["reward_identity_error"]) <= 1e-8
            for row in rows
        )
    )
    return {
        "rows": rows,
        "aggregate": {
            "random_mean_jct": random_mean,
            "learned_mean_jct": learned_mean,
            "greedy_mean_jct": greedy_mean,
            "srw_mean_jct": srw_mean,
            "learned_improvement_over_random": improvement,
            "required_improvement": config.minimum_random_improvement,
            "gate_passed": gate_passed,
        },
    }


def _save_learning_curve(
    raw_returns: Sequence[float],
    output: Path,
    title: str = "Toy MaskablePPO Training",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required for an RL curve") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [-value for value in raw_returns]
    window = min(50, max(1, len(values)))
    moving = [
        fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(values, color="#A9B4C2", linewidth=0.8, alpha=0.55)
    axis.plot(
        moving,
        color="#2364AA",
        linewidth=2.0,
        label=f"{window}-episode moving mean",
    )
    axis.set_title(title)
    axis.set_xlabel("Completed episode")
    axis.set_ylabel("Total workflow JCT (lower is better)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train_toy_policy(
    config: ToyTrainingConfig,
    model_path: Path,
    output_path: Path,
    plot_path: Path | None,
) -> dict[str, Any]:
    env = make_toy_env(config)
    callback = EpisodeHistoryCallback()
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        seed=config.seed,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.entropy_coefficient,
        verbose=0,
        device="cpu",
    )
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    evaluation = evaluate_toy_model(model, config)
    payload = {
        "phase": "5C-toy-learning-gate",
        "algorithm": "MaskablePPO",
        "policy": "MultiInputPolicy",
        "config": asdict(config),
        "model_path": str(model_path),
        "completed_training_episodes": len(callback.raw_returns),
        "training_raw_returns": callback.raw_returns,
        "evaluation": evaluation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if plot_path is not None:
        _save_learning_curve(callback.raw_returns, plot_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and validate the Phase 5 toy MaskablePPO scheduler."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_training_config(args.config)
        payload = train_toy_policy(
            config,
            args.model,
            args.output,
            args.plot,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    aggregate = payload["evaluation"]["aggregate"]
    print(
        f"Toy held-out Mean JCT: random={aggregate['random_mean_jct']:.3f} "
        f"| learned={aggregate['learned_mean_jct']:.3f} "
        f"| greedy={aggregate['greedy_mean_jct']:.3f} "
        f"| SRW={aggregate['srw_mean_jct']:.3f}"
    )
    print(
        "Improvement over random: "
        f"{aggregate['learned_improvement_over_random']:.1%} "
        f"(required {aggregate['required_improvement']:.1%})"
    )
    print(f"Learning gate passed: {aggregate['gate_passed']}")
    print(f"Model: {args.model}")
    print(f"JSON: {args.output}")
    print(f"Plot: {args.plot}")
    return 0 if aggregate["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
