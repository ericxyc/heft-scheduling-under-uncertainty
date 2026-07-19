"""Train MaskablePPO on mixed small WfCommons workflow traces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from random import Random
from typing import Sequence

from sb3_contrib import MaskablePPO

from ..benchmark_data import BenchmarkCorpus, load_benchmark_corpus
from ..dynamic_benchmark import mean_interarrival_for_load
from ..dynamic_scenario import build_dynamic_scenario
from ..trace_model import load_trace_model_config
from .environment import DynamicSchedulingEnv
from .training import EpisodeHistoryCallback, _save_learning_curve


DEFAULT_CONFIG = Path("configs/rl/wfcommons_maskable_ppo.json")
DEFAULT_MANIFEST = Path("configs/workflow_benchmark.json")
DEFAULT_WORKER_CONFIG = Path("configs/trace_worker_model.json")
DEFAULT_MODEL = Path(
    "artifacts/rl/final_models/wfcommons_maskable_ppo.zip"
)
DEFAULT_OUTPUT = Path("results/rl/wfcommons_training.json")
DEFAULT_PLOT = Path("results/rl/wfcommons_learning_curve.png")


@dataclass(frozen=True)
class WfCommonsTrainingConfig:
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
    training_loads: tuple[float, ...]
    training_cvs: tuple[float, ...]

    def validate(self) -> None:
        counts = (
            self.total_timesteps,
            self.n_steps,
            self.batch_size,
            self.n_epochs,
            self.max_candidates,
            self.workflow_count,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("positive training counts must be greater than zero")
        if not self.training_loads or not self.training_cvs:
            raise ValueError("training loads and CVs must be non-empty")
        if any(
            not isfinite(value) or value <= 0
            for value in self.training_loads
        ):
            raise ValueError("training loads must be positive and finite")
        if any(
            not isfinite(value) or value < 0
            for value in self.training_cvs
        ):
            raise ValueError("training CVs must be non-negative and finite")
        if self.learning_rate <= 0 or self.reward_scale <= 0:
            raise ValueError("learning rate and reward scale must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and GAE lambda must be in [0, 1]")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy coefficient must be non-negative")


def load_wf_training_config(path: Path) -> WfCommonsTrainingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("WfCommons RL config must be a JSON object")
    payload["training_loads"] = tuple(payload.get("training_loads", ()))
    payload["training_cvs"] = tuple(payload.get("training_cvs", ()))
    expected = set(WfCommonsTrainingConfig.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(
            f"WfCommons RL config fields must be exactly {sorted(expected)}"
        )
    config = WfCommonsTrainingConfig(**payload)
    config.validate()
    return config


def make_wfcommons_env(
    config: WfCommonsTrainingConfig,
    corpus: BenchmarkCorpus,
) -> DynamicSchedulingEnv:
    templates = corpus.templates_for_size("small")

    def scenario_factory(seed: int):
        generator = Random(seed)
        offered_load = config.training_loads[
            generator.randrange(len(config.training_loads))
        ]
        runtime_cv = config.training_cvs[
            generator.randrange(len(config.training_cvs))
        ]
        mean_interarrival = mean_interarrival_for_load(
            templates,
            config.workflow_count,
            offered_load,
        )
        return build_dynamic_scenario(
            templates=templates,
            workflow_count=config.workflow_count,
            mean_interarrival_time=mean_interarrival,
            runtime_cv=runtime_cv,
            seed=seed,
        )

    return DynamicSchedulingEnv(
        scenario_factory=scenario_factory,
        processors=templates[0].workflow.processors,
        max_candidates=config.max_candidates,
        reward_scale=config.reward_scale,
    )


def train_wfcommons_policy(
    config: WfCommonsTrainingConfig,
    corpus: BenchmarkCorpus,
    model_path: Path,
    output_path: Path,
    plot_path: Path | None,
) -> dict[str, object]:
    env = make_wfcommons_env(config, corpus)
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
    payload: dict[str, object] = {
        "phase": "5D-wfcommons-training",
        "algorithm": "MaskablePPO",
        "policy": "MultiInputPolicy",
        "config": {
            **asdict(config),
            "training_loads": list(config.training_loads),
            "training_cvs": list(config.training_cvs),
        },
        "corpus_name": corpus.name,
        "training_families": sorted(
            {entry.family for entry in corpus.entries if entry.size == "small"}
        ),
        "model_path": str(model_path),
        "completed_training_episodes": len(callback.raw_returns),
        "training_raw_returns": callback.raw_returns,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if plot_path is not None:
        _save_learning_curve(
            callback.raw_returns,
            plot_path,
            title="WfCommons Candidate MaskablePPO Training",
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO on mixed small WfCommons traces."
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
        config = load_wf_training_config(args.config)
        worker_config = load_trace_model_config(args.worker_config)
        corpus = load_benchmark_corpus(args.manifest, worker_config)
        payload = train_wfcommons_policy(
            config,
            corpus,
            args.model,
            args.output,
            args.plot,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    print(
        f"Completed episodes: {payload['completed_training_episodes']} | "
        f"families={payload['training_families']}"
    )
    print(f"Model: {args.model}")
    print(f"JSON: {args.output}")
    print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
