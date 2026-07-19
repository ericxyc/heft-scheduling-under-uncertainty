## Context

The HEFT paper publishes a deterministic 10-task example with one DAG, 15
communication-cost edges, three heterogeneous processors, and a computation
cost for every task-processor pair. Its HEFT schedule has a makespan of 80.

This project has no existing scheduler code. The first implementation must be
small enough to audit against the paper and reusable by later WfCommons,
uncertainty, and online-scheduling experiments.

## Goals / Non-Goals

**Goals:**

- Represent the workflow and schedule with typed, immutable-friendly Python
  data structures.
- Reproduce the paper's upward ranks, priority order, processor assignments,
  start/finish times, and makespan.
- Implement the insertion policy rather than an append-only approximation.
- Validate schedule feasibility independently of the scheduling procedure.
- Keep the core scheduler free of plotting, file-format, and RL dependencies.

**Non-Goals:**

- Reproduce all 56,250 random-DAG experiments.
- Implement CPOP, DLS, LMT, or other paper baselines.
- Parse Pegasus/WfCommons data.
- Model runtime uncertainty, dynamic arrivals, GPU memory, or energy.
- Claim optimality; HEFT is a heuristic.

## Decisions

### Use a small standard-library Python package

The scheduler and tests will use dataclasses, mappings, JSON, argparse, and
unittest. This keeps the correctness benchmark runnable without network access.

Alternative considered: NetworkX. It is useful later for dataset analysis, but
the paper fixture is small and a custom model makes dependency semantics and
topological validation explicit.

### Separate candidate finish time from committed finish time

Processor selection will evaluate a candidate earliest start/finish time for
each processor. Only the selected candidate becomes a committed
`ScheduleEntry`. This preserves the distinction between EFT and AFT used in the
paper.

### Use processor timelines and first-fit insertion

Each processor stores schedule entries sorted by start time. Candidate
placement scans gaps from time zero and selects the first gap that begins after
all predecessor data are ready and can contain the task's computation cost.

Alternative considered: append every task after the processor's last job. That
is simpler but does not reproduce HEFT's insertion-based policy.

### Keep paper data in one reviewed fixture

The Figure 3 DAG, communication costs, and computation matrix will live in
`paper_example.py`. Expected ranks and task order will be stored beside the
fixture so transcription errors are visible during review.

### Validate schedules independently

The validator will check task coverage, duplicate assignment, processor
overlap, duration, precedence, and cross-processor communication delays. Tests
will call the validator on both valid and intentionally invalid schedules.

### Produce JSON as the canonical result artifact

The CLI will print a readable table and write JSON. A separate visualization
module will consume the same schedule and optionally use Matplotlib to write a
PNG. Plotting cannot affect scheduling results.

## Risks / Trade-offs

- **Paper transcription error** -> Keep all values in one fixture and compare
  computed ranks and makespan with published values.
- **Floating-point tie behavior** -> Use deterministic task IDs as a final
  tie-breaker and compare ranks with a tolerance.
- **Insertion-policy edge case** -> Test a workflow where a task fits only in
  an internal processor gap.
- **Visualization dependency missing** -> Keep PNG generation optional; core
  reproduction and tests remain standard-library only.
- **Later simulator needs richer models** -> Add future fields through new
  types rather than weakening Phase 1 invariants.

## Migration Plan

This is a new standalone project. No migration or rollback is required.

## Open Questions

- Later phases must decide whether WfCommons task runtimes define estimated
  costs, actual costs, or distributions fitted by task type.
- The RL phase must select an action representation only after dynamic
  heuristic baselines are working.
