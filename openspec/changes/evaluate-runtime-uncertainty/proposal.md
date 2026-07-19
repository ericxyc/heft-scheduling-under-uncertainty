# Evaluate Runtime Uncertainty

## Why

The current trace-driven baseline assumes that HEFT's runtime estimates are
exact. Real workflow execution times vary, so the project needs a controlled
experiment that measures the effect of estimate error before introducing an
online or learning-based scheduler.

## What Changes

- Add deterministic log-normal task-duration perturbations with a specified
  coefficient of variation (CV).
- Replay a static HEFT plan under perturbed durations while preserving its
  worker assignment and planned per-worker task order.
- Compare that replay with a perfect-information HEFT reference that receives
  the perturbed durations before scheduling.
- Add a CLI, JSON report, summary plot, documentation, and tests.

## Non-goals

- This change does not implement online rescheduling, dynamic workflow
  arrivals, GPU measurement, or reinforcement learning.
- The perfect-information HEFT reference is not an optimal oracle and is not a
  deployable policy.
