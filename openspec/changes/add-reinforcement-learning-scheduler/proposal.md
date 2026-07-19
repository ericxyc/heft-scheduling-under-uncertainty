# Add Reinforcement-Learning Scheduler

## Why

Phase 4B exposes a state-dependent tradeoff between critical-path priority,
workflow aging, and shortest remaining work. A learned policy can test whether
those signals can be combined adaptively under dynamic arrivals and runtime
uncertainty. The experiment must preserve the existing simulator and strong
heuristic baselines so a negative RL result remains meaningful.

## What Changes

- Refactor the event simulator into a pausable decision core while preserving
  the existing five policy results.
- Add a Gymnasium environment with fixed-shape observations, masked discrete
  task-worker actions, and a dense reward exactly aligned with total workflow
  JCT.
- Add random and heuristic environment adapters for correctness checks.
- Add optional PyTorch, Stable-Baselines3, and MaskablePPO training support.
- Train a small reproducible policy, require a held-out improvement over random,
  and compare a frozen policy with all Phase 4B heuristics.
- Record configurations, seeds, checkpoints, learning curves, evaluation
  metrics, and limitations.

## Non-goals

- RL is optional and SHALL NOT be required to run HEFT or Phase 4 benchmarks.
- The first policy uses tabular candidate features and an MLP, not a graph
  neural network.
- A local training run does not establish production GPU-cluster performance.
- The implementation does not claim RL superiority unless held-out paired
  evaluation supports it.
