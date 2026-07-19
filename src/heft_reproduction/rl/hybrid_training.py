"""Train the V2 RL policy that selects among heuristic recommendations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
from typing import Sequence

from sb3_contrib import MaskablePPO

from ..benchmark_data import load_benchmark_corpus
from ..dynamic_benchmark import mean_interarrival_for_load
from ..dynamic_scenario import build_dynamic_scenario
from ..trace_model import load_trace_model_config
from .hybrid_environment import HeuristicSelectionEnv
from .training import EpisodeHistoryCallback, _save_learning_curve
from .wfcommons_training import (
    WfCommonsTrainingConfig,
    load_wf_training_config,
)


DEFAULT_CONFIG = Path(
    "configs/rl/wfcommons_hybrid_maskable_ppo.json"
)
DEFAULT_MANIFEST = Path("configs/workflow_benchmark.json")
DEFAULT_WORKER_CONFIG = Path("configs/trace_worker_model.json")
DEFAULT_MODEL = Path(
    "artifacts/rl/final_models/wfcommons_hybrid_maskable_ppo.zip"
)
DEFAULT_OUTPUT = Path("results/rl/wfcommons_hybrid_training.json")
DEFAULT_PLOT = Path("results/rl/wfcommons_hybrid_learning_curve.png")


def make_hybrid_env(config: WfCommonsTrainingConfig, corpus):
    templates = corpus.templates_for_size("small")

    def scenario_factory(seed: int):
        generator = Random(seed)
        load = config.training_loads[
            generator.randrange(len(config.training_loads))
        ]
        cv = config.training_cvs[
            generator.randrange(len(config.training_cvs))
        ]
        return build_dynamic_scenario(
            templates,
            config.workflow_count,
            mean_interarrival_for_load(
                templates,
                config.workflow_count,
                load,
            ),
            cv,
            seed,
        )

    return HeuristicSelectionEnv(
        scenario_factory=scenario_factory,
        processors=templates[0].workflow.processors,
        max_candidates=config.max_candidates,
        reward_scale=config.reward_scale,
    )


def train_hybrid_policy(
    config: WfCommonsTrainingConfig,
    corpus,
    model_path: Path,
    output_path: Path,
    plot_path: Path | None,
):
    env = make_hybrid_env(config, corpus)
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
    payload = {
        "phase": "5D-hybrid-wfcommons-training",
        "algorithm": "MaskablePPO",
        "environment": "heuristic-selection",
        "config": {
            **asdict(config),
            "training_loads": list(config.training_loads),
            "training_cvs": list(config.training_cvs),
        },
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
            title="WfCommons Hybrid MaskablePPO Training",
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO to select among five heuristic proposals."
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
        corpus = load_benchmark_corpus(
            args.manifest,
            load_trace_model_config(args.worker_config),
        )
        payload = train_hybrid_policy(
            config,
            corpus,
            args.model,
            args.output,
            args.plot,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 2
    print(f"Completed episodes: {payload['completed_training_episodes']}")
    print(f"Model: {args.model}")
    print(f"JSON: {args.output}")
    print(f"Plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
