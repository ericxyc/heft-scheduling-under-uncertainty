"""Train a DAG-aware MaskablePPO scheduler on WfCommons workflows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter
from typing import Sequence

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from torch import nn

from ..benchmark_data import BenchmarkCorpus, load_benchmark_corpus
from ..dynamic_benchmark import mean_interarrival_for_load
from ..dynamic_scenario import build_dynamic_scenario
from ..trace_model import load_trace_model_config
from .graph_environment import GraphHeuristicSelectionEnv, GraphSchedulingEnv
from .graph_policy import GraphCandidateExtractor
from .training import EpisodeHistoryCallback


DEFAULT_CONFIG = Path("configs/rl/wfcommons_gnn_ppo.json")
DEFAULT_MANIFEST = Path("configs/workflow_benchmark.json")
DEFAULT_WORKER_CONFIG = Path("configs/trace_worker_model.json")
DEFAULT_MODEL = Path("artifacts/rl/final_models/wfcommons_gnn_ppo.zip")
DEFAULT_OUTPUT = Path("results/rl/wfcommons_gnn_training.json")
DEFAULT_PLOT = Path("results/rl/wfcommons_gnn_training.png")


@dataclass(frozen=True)
class GNNTrainingConfig:
    seed: int
    device: str
    policy_mode: str
    greedy_prior_logit: float
    total_timesteps: int
    max_training_seconds: float
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    entropy_coefficient: float
    entropy_coefficient_final: float
    entropy_decay_fraction: float
    max_candidates: int
    max_nodes: int
    max_edges: int
    reward_scale: float
    potential_weight: float
    workflow_count: int
    training_loads: tuple[float, ...]
    training_cvs: tuple[float, ...]
    hidden_dim: int
    candidate_dim: int
    message_passing_steps: int

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.policy_mode not in {"candidate", "heuristic"}:
            raise ValueError("policy mode must be candidate or heuristic")
        if not isfinite(self.greedy_prior_logit) or self.greedy_prior_logit < 0:
            raise ValueError("greedy prior logit must be non-negative and finite")
        counts = (
            self.total_timesteps,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.max_candidates,
            self.max_nodes,
            self.max_edges,
            self.workflow_count,
            self.hidden_dim,
            self.candidate_dim,
            self.message_passing_steps,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("all training and graph counts must be positive")
        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size")
        if not self.training_loads or not self.training_cvs:
            raise ValueError("training loads and CVs must be non-empty")
        if any(not isfinite(value) or value <= 0 for value in self.training_loads):
            raise ValueError("training loads must be positive and finite")
        if any(not isfinite(value) or value < 0 for value in self.training_cvs):
            raise ValueError("training CVs must be non-negative and finite")
        if self.learning_rate <= 0 or self.reward_scale <= 0:
            raise ValueError("learning rate and reward scale must be positive")
        entropy_values = (
            self.entropy_coefficient,
            self.entropy_coefficient_final,
            self.entropy_decay_fraction,
        )
        if any(not isfinite(value) for value in entropy_values):
            raise ValueError("entropy schedule values must be finite")
        if self.entropy_coefficient < 0 or self.entropy_coefficient_final < 0:
            raise ValueError("entropy coefficients must be non-negative")
        if self.entropy_coefficient_final > self.entropy_coefficient:
            raise ValueError(
                "final entropy coefficient must not exceed the initial value"
            )
        if not 0 < self.entropy_decay_fraction <= 1:
            raise ValueError("entropy decay fraction must be in (0, 1]")
        if not isfinite(self.max_training_seconds) or self.max_training_seconds <= 0:
            raise ValueError("training time budget must be positive and finite")
        if self.potential_weight < 0:
            raise ValueError("potential weight must be non-negative")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and GAE lambda must be in [0, 1]")


def load_gnn_training_config(path: Path) -> GNNTrainingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GNN training config must be a JSON object")
    payload.setdefault(
        "entropy_coefficient_final",
        payload.get("entropy_coefficient"),
    )
    payload.setdefault("entropy_decay_fraction", 1.0)
    payload["training_loads"] = tuple(payload.get("training_loads", ()))
    payload["training_cvs"] = tuple(payload.get("training_cvs", ()))
    expected = set(GNNTrainingConfig.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(
            f"GNN training config fields must be exactly {sorted(expected)}"
        )
    config = GNNTrainingConfig(**payload)
    config.validate()
    return config


def linear_entropy_coefficient(
    initial: float,
    final: float,
    completed_fraction: float,
    decay_fraction: float,
) -> float:
    """Linearly decay entropy, reaching the floor within the chosen budget."""

    progress = min(max(completed_fraction / decay_fraction, 0.0), 1.0)
    return initial + progress * (final - initial)


class LinearEntropyScheduleCallback(BaseCallback):
    """Update MaskablePPO's entropy coefficient between rollout updates."""

    def __init__(
        self,
        initial: float,
        final: float,
        decay_fraction: float,
        total_timesteps: int,
    ) -> None:
        super().__init__(verbose=0)
        self.initial = initial
        self.final = final
        self.decay_fraction = decay_fraction
        self.total_timesteps = total_timesteps
        self.history: list[dict[str, float]] = []

    def _set_coefficient(self, completed_fraction: float) -> None:
        coefficient = linear_entropy_coefficient(
            self.initial,
            self.final,
            completed_fraction,
            self.decay_fraction,
        )
        self.model.ent_coef = coefficient
        row = {
            "timesteps": float(self.num_timesteps),
            "entropy_coefficient": coefficient,
        }
        if self.history and self.history[-1]["timesteps"] == row["timesteps"]:
            self.history[-1] = row
        else:
            self.history.append(row)

    def _on_training_start(self) -> None:
        self._set_coefficient(0.0)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._set_coefficient(self.num_timesteps / self.total_timesteps)


def make_gnn_env(config: GNNTrainingConfig, corpus: BenchmarkCorpus):
    templates = corpus.templates_for_size("small")

    def scenario_factory(seed: int):
        generator = Random(seed)
        load = config.training_loads[generator.randrange(len(config.training_loads))]
        cv = config.training_cvs[generator.randrange(len(config.training_cvs))]
        return build_dynamic_scenario(
            templates=templates,
            workflow_count=config.workflow_count,
            mean_interarrival_time=mean_interarrival_for_load(
                templates, config.workflow_count, load
            ),
            runtime_cv=cv,
            seed=seed,
        )

    environment_class = (
        GraphHeuristicSelectionEnv
        if config.policy_mode == "heuristic"
        else GraphSchedulingEnv
    )
    return environment_class(
        scenario_factory=scenario_factory,
        processors=templates[0].workflow.processors,
        max_candidates=config.max_candidates,
        max_nodes=config.max_nodes,
        max_edges=config.max_edges,
        reward_scale=config.reward_scale,
        reward_mode="jct-progress-potential",
        potential_weight=config.potential_weight,
    )


class BudgetedEpisodeHistoryCallback(EpisodeHistoryCallback):
    """Collect diagnostics and stop cleanly at a wall-clock budget."""

    def __init__(self, max_seconds: float) -> None:
        super().__init__()
        self.max_seconds = max_seconds
        self.started = 0.0
        self.stopped_by_time_budget = False
        self.heuristic_action_counts: dict[str, int] = {}

    def _on_training_start(self) -> None:
        self.started = perf_counter()

    def _on_step(self) -> bool:
        if not super()._on_step():
            return False
        for done, info in zip(
            self.locals.get("dones", ()),
            self.locals.get("infos", ()),
        ):
            counts = info.get("heuristic_action_counts") if done else None
            if isinstance(counts, dict):
                for policy, count in counts.items():
                    self.heuristic_action_counts[policy] = (
                        self.heuristic_action_counts.get(policy, 0) + int(count)
                    )
        if perf_counter() - self.started >= self.max_seconds:
            self.stopped_by_time_budget = True
            return False
        return True

    def _capture_diagnostics(self) -> None:
        previous_count = len(self.diagnostics)
        super()._capture_diagnostics()
        if len(self.diagnostics) > previous_count:
            self.diagnostics[-1]["entropy_coefficient"] = float(
                self.model.ent_coef
            )


def _save_training_plot(callback: EpisodeHistoryCallback, output: Path) -> None:
    import matplotlib.pyplot as plt

    objective = [-value for value in callback.raw_returns]
    window = min(25, max(1, len(objective)))
    moving = [
        fmean(objective[max(0, index - window + 1) : index + 1])
        for index in range(len(objective))
    ]
    steps = [row["timesteps"] for row in callback.diagnostics]
    figure, axes = plt.subplots(4, 1, figsize=(10, 14))
    axes[0].plot(objective, alpha=0.25, color="#7F8C8D")
    axes[0].plot(moving, color="#1F77B4", linewidth=2)
    axes[0].set_ylabel("Total workflow JCT")
    axes[0].set_title("GNN+PPO training episode objective (lower is better)")
    axes[1].plot(callback.training_returns, color="#2CA02C")
    axes[1].set_ylabel("Shaped episode return")
    axes[1].set_title("Potential-shaped training reward")
    entropy = [
        -row.get("entropy_loss", float("nan"))
        for row in callback.diagnostics
    ]
    coefficients = [
        row.get("entropy_coefficient", float("nan"))
        for row in callback.diagnostics
    ]
    axes[2].plot(steps, entropy, label="policy_entropy", color="#17BECF")
    coefficient_axis = axes[2].twinx()
    coefficient_axis.plot(
        steps,
        coefficients,
        label="entropy_coefficient",
        color="#8C564B",
    )
    axes[2].set_ylabel("Policy entropy")
    coefficient_axis.set_ylabel("Entropy coefficient")
    axes[2].set_title("Exploration diagnostics")
    lines = axes[2].lines + coefficient_axis.lines
    axes[2].legend(lines, [line.get_label() for line in lines])
    for key, color in (
        ("policy_gradient_loss", "#9467BD"),
        ("value_loss", "#D62728"),
        ("loss", "#FF7F0E"),
    ):
        values = [row.get(key, float("nan")) for row in callback.diagnostics]
        axes[3].plot(steps, values, label=key, color=color)
    axes[3].set_xlabel("Timesteps")
    axes[3].set_ylabel("Loss")
    axes[3].set_title("PPO optimization diagnostics")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train_gnn_policy(
    config: GNNTrainingConfig,
    corpus: BenchmarkCorpus,
    model_path: Path,
    output_path: Path,
    plot_path: Path | None,
) -> dict[str, object]:
    env = make_gnn_env(config, corpus)
    callback = BudgetedEpisodeHistoryCallback(config.max_training_seconds)
    entropy_schedule = LinearEntropyScheduleCallback(
        initial=config.entropy_coefficient,
        final=config.entropy_coefficient_final,
        decay_fraction=config.entropy_decay_fraction,
        total_timesteps=config.total_timesteps,
    )
    checkpoint = CheckpointCallback(
        save_freq=5000,
        save_path=str(model_path.parent / "checkpoints"),
        name_prefix=model_path.stem,
    )
    policy_kwargs = {
        "features_extractor_class": GraphCandidateExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": config.hidden_dim,
            "candidate_dim": config.candidate_dim,
            "message_passing_steps": config.message_passing_steps,
        },
        "net_arch": {"pi": [256, 128], "vf": [128, 64]},
        "activation_fn": nn.SiLU,
        "ortho_init": False,
    }
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.entropy_coefficient,
        target_kl=0.03,
        verbose=1,
        device=config.device,
    )
    if config.policy_mode == "heuristic" and config.greedy_prior_logit > 0:
        with torch.no_grad():
            model.policy.action_net.bias[0] = config.greedy_prior_logit
    started = perf_counter()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList([callback, entropy_schedule, checkpoint]),
        progress_bar=False,
    )
    elapsed = perf_counter() - started
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    payload: dict[str, object] = {
        "phase": "6-gnn-maskable-ppo",
        "algorithm": "MaskablePPO",
        "reward": "JCT plus policy-invariant DAG-progress potential",
        "config": {
            **asdict(config),
            "training_loads": list(config.training_loads),
            "training_cvs": list(config.training_cvs),
        },
        "training_seconds": elapsed,
        "completed_training_episodes": len(callback.raw_returns),
        "completed_timesteps": model.num_timesteps,
        "stopped_by_time_budget": callback.stopped_by_time_budget,
        "raw_episode_returns": callback.raw_returns,
        "shaped_episode_returns": callback.training_returns,
        "exploration": {
            "greedy_prior_logit": config.greedy_prior_logit,
            "entropy_schedule": {
                "kind": "linear",
                "initial": config.entropy_coefficient,
                "final": config.entropy_coefficient_final,
                "decay_fraction": config.entropy_decay_fraction,
                "history": entropy_schedule.history,
            },
            "training_heuristic_action_counts": (
                callback.heuristic_action_counts
            ),
        },
        "diagnostics": callback.diagnostics,
        "model_path": str(model_path),
        "device": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            ),
        },
        "training_families": sorted(
            {entry.family for entry in corpus.entries if entry.size == "small"}
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if plot_path is not None:
        _save_training_plot(callback, plot_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DAG-aware MaskablePPO")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--worker-config", type=Path, default=DEFAULT_WORKER_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--timesteps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_gnn_training_config(args.config)
        if args.timesteps is not None:
            config = replace(config, total_timesteps=args.timesteps)
            config.validate()
        worker_config = load_trace_model_config(args.worker_config)
        corpus = load_benchmark_corpus(args.manifest, worker_config)
        payload = train_gnn_policy(
            config, corpus, args.model, args.output, args.plot
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    print(
        f"Episodes: {payload['completed_training_episodes']} | "
        f"seconds: {payload['training_seconds']:.1f}"
    )
    print(f"Model: {args.model}")
    print(f"JSON: {args.output}")
    print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
