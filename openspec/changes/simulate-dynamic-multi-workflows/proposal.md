# Simulate Dynamic Multi-Workflow Scheduling

## Why

The current experiments contain one fully known workflow and one static
schedule. They cannot measure queueing, cross-workflow contention, job
completion time, or the benefit of replanning when workflows arrive over time.
An event-driven online simulator and transparent heuristic baselines are needed
before reinforcement learning can be evaluated fairly.

## What Changes

- Add deterministic dynamic scenarios containing complete workflow DAGs that
  arrive over time.
- Add a non-preemptive event-driven simulator with dependency, communication,
  worker-availability, and runtime-uncertainty semantics.
- Add three baseline policies: online greedy earliest finish, per-workflow
  static HEFT, and rolling HEFT.
- Report workflow JCT, P95 JCT, queue wait, makespan, throughput, utilization,
  transfer volume, decision count, candidate evaluations, and measured
  scheduler wall time.
- Add a CLI, deterministic core JSON report, comparison visualization, tests,
  and documentation.

## Non-goals

- This phase does not implement reinforcement learning, task preemption,
  migration, elastic GPU allocation, failures, or unknown task graphs.
- Repeated copies of the vendored Montage trace validate the simulator but do
  not establish cross-application generalization.
