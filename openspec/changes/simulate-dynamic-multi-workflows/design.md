# Design: Dynamic Multi-Workflow Scheduling

## Scenario

Each workflow instance arrives as a complete DAG at a seeded arrival time.
Before arrival, the scheduler cannot observe the instance. After arrival, only
tasks whose parents have completed are dependency-ready. Actual task durations
use the Phase 3 task-level mean-one log-normal uncertainty model.

The first scenario builder supports one or more WfFormat-derived templates. It
selects templates round-robin and samples exponential inter-arrival times from
a configured mean. An inter-arrival mean of zero places all workflows at time
zero for deterministic contention tests.

## Event Model

The simulator is non-preemptive. Its event clock advances to the earliest of:

1. a workflow arrival;
2. a running task completion;
3. an inter-worker input transfer becoming available to an idle worker.

At an event, all simultaneous completions and arrivals are processed before
new decisions. A running task keeps its worker until completion. A task may
start only when all parents have actually completed, its inputs are available
on the chosen worker, and that worker is idle.

## Policies

### Online Greedy EFT

Among currently feasible task-worker pairs, choose the pair with the smallest
estimated finish time. Ties use workflow arrival, instance ID, local task ID,
and configured worker order.

### Per-Workflow Static HEFT

Run HEFT once for each workflow template using estimated durations. Preserve
the resulting worker assignment and within-workflow per-worker order. When
multiple workflow instances want the same worker, use the earliest absolute
planned start followed by deterministic instance/task tie-breaking. Running
and unstarted tasks are never reassigned.

### Rolling HEFT

Use each arrived workflow's HEFT upward rank to prioritize currently feasible
ready tasks across instances. For the chosen task, select the feasible idle
worker with the smallest estimated finish time. Re-evaluate after every event;
only started tasks are committed.

This is a transparent online HEFT baseline, not a claim of a novel heuristic.

## Fairness and Reproducibility

All policies consume the same immutable scenario, including arrival times and
actual duration multipliers. Simulated schedules and logical metrics are
deterministic for fixed inputs and seeds. Measured scheduler wall time is
machine-dependent and is separated from deterministic candidate-evaluation
counts.

## Metrics

Workflow JCT is completion minus arrival. Queue wait is task start minus the
time its inputs become available on its assigned worker. P95 uses linear
interpolation between sorted samples. Throughput is completed workflows divided
by the simulation horizon. Utilization is actual busy time divided by horizon.
