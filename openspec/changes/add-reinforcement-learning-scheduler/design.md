# Design: Phase 5 Reinforcement Learning

## Simulator Boundary

`DynamicSchedulingCore` owns immutable scenario data and mutable event state.
It exposes event processing, feasible candidate generation, one non-preemptive
task commitment, clock advancement, and final validation. Existing heuristic
simulation remains a wrapper around this core.

The RL environment controls only candidate selection. It SHALL use the same
runtime realizations, communication rules, event transitions, and validator as
the heuristic policies.

## Observation

The environment emits a fixed-size dictionary containing:

- one row per candidate slot with estimated finish time, estimated duration,
  upward rank, workflow age, remaining work, completion fraction, task graph
  features, and worker identity/state;
- global load, time, active-workflow, ready-task, and worker-busy features;
- a binary valid-action mask.

Continuous values are normalized by per-scenario reference scales. Sampled
actual task duration is never observable before completion.

## Action

The action is a discrete candidate slot representing one feasible
`(workflow task, worker)` pair. Unused slots are invalid. If feasible candidates
exceed the configured capacity, deterministic truncation retains a diverse
shortlist and reports the truncation count. The environment SHALL reject an
invalid action even when training normally masks it.

## Reward

For every simulated interval:

```text
reward = -active_arrived_incomplete_workflows * elapsed_time
```

The undiscounted episode return therefore equals negative total workflow JCT.
The implementation MAY normalize rewards for optimization, but SHALL report
the raw objective and verify the identity at episode completion.

## Training Gates

1. Heuristic adapters reproduce direct-simulator schedules and logical metrics.
2. Random masked actions complete every task without validation errors.
3. A small MaskablePPO policy improves held-out mean JCT over random by at
   least 10% across a seeded toy evaluation set.
4. A frozen policy is evaluated on shared WfCommons scenarios against all five
   heuristic baselines.

Failure of a gate triggers diagnosis or a documented policy iteration. It does
not remove or alter the heuristic scheduling path.

## Evaluation

Training and evaluation seeds are disjoint. The primary metric is paired mean
JCT. Reports also include P95 JCT, task wait, utilization, communication,
decision count, inference wall time, and schedule validity. Small traces are
used for training iteration; medium traces remain held-out scale checks.
