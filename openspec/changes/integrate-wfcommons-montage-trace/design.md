## Context

Phase 1 exposes a validated `Workflow` model and deterministic insertion-based
HEFT scheduler for integer task IDs. The selected WfCommons instance uses
WfFormat 1.5 and contains separate specification and execution sections:
58 task specifications, 58 execution records, 114 dependency edges, 111 files,
one observed machine, and an observed makespan of 1060 seconds.

The trace contains one observed runtime per task, not a counterfactual runtime
matrix for several machines. File sizes are measured trace data, while worker
heterogeneity and network bandwidth must therefore be explicit simulation
assumptions.

## Goals / Non-Goals

**Goals:**

- Preserve and validate the raw trace before transforming it.
- Join specification and execution tasks by source ID.
- Reuse the existing HEFT scheduler without weakening its validation.
- Derive processor-specific runtimes and edge communication costs with a
  deterministic, documented configuration.
- Export enough source and modeling metadata to reproduce every result.
- Keep the paper reproduction command and tests unchanged.
- Make the Pegasus workflow-definition API available without requiring the full
  planner or HTCondor for trace experiments.

**Non-Goals:**

- Claim that simulated workers represent measured CPUs or GPUs.
- Reproduce the original Pegasus schedule or compare makespans as if the
  execution environments were identical.
- Model memory capacity, multicore execution, failures, dynamic arrivals, or
  runtime uncertainty in this phase.
- Add the WfCommons Python package as a required runtime dependency.

## Decisions

### Vendor one immutable raw instance

Store the official Montage `2mass-005d` JSON unchanged under `data/raw/` and
record its URL, Git blob SHA, SHA-256, WfFormat version, and upstream license.
This keeps tests and experiments offline and protects against upstream changes.

Alternative: download on every run. Rejected because it makes experiments
network-dependent and less reproducible.

### Keep Pegasus integration optional

Pin `pegasus-wms.api==5.1.2` in a `pegasus` dependency extra. This makes
`Pegasus.api.Workflow`, `Job`, `Transformation`, and `File` available for future
workflow-definition exercises while keeping the current WfFormat parser based
on the Python standard library. The full planner and HTCondor remain outside
this phase because no real workflow submission is required.

### Parse with typed local data structures

Use `json` plus frozen dataclasses for files, task specifications, execution
records, and the joined trace. Validate required sections, unique IDs, matching
specification/execution task sets, parent/child consistency, file references,
positive runtimes, and acyclicity before conversion.

Alternative: depend directly on `wfcommons`. Deferred because Phase 2 only
needs a small stable schema subset and the current project otherwise has no core
dependencies.

### Preserve source identity through a stable ID map

Sort WfFormat task IDs and assign deterministic integer IDs required by the
existing `Workflow`. Store both directions and task metadata in a
`ModeledTraceWorkflow` wrapper so reports never lose the original task ID or
program name.

### Use an explicit CPU/I/O proxy worker model

The balanced worker reproduces each observed runtime. Two additional simulated
workers trade compute and I/O speed:

- `Compute-Fast`: compute speed 1.6, I/O speed 0.8
- `Balanced`: compute speed 1.0, I/O speed 1.0
- `IO-Fast`: compute speed 0.8, I/O speed 1.6

Treat `avgCPU / 100` as a bounded CPU-weight proxy and calculate:

`modeled_runtime = observed_runtime * (cpu_weight / compute_speed + (1 - cpu_weight) / io_speed)`

This is a transparent sensitivity model, not a hardware performance claim.
Missing `avgCPU` values use a configured default weight.

Alternative: apply one scalar speed per worker. Rejected because every task
would prefer the same worker and task heterogeneity would be uninformative.

### Derive communication from shared files

For every declared dependency, sum the sizes of files in the intersection of
the parent outputs and child inputs. Communication cost is shared bytes divided
by configured bandwidth when tasks use different workers; the existing
scheduler already makes same-worker communication zero. Preserve zero-cost
dependency edges if a future trace has no shared-file match.

### Keep trace reporting separate from the paper CLI

Add `heft-wfcommons` with input, worker-config, JSON-output, and optional plot
arguments. Report trace summary, model assumptions, source task labels,
processor utilization, cross-worker bytes, simulated serial baseline, speedup,
schedule validity, and modeled makespan.

## Risks / Trade-offs

- [Risk] `avgCPU` is not a physically exact decomposition of runtime. →
  Describe it as a proxy and serialize the formula and parameters.
- [Risk] Observed Pegasus makespan includes queueing, orchestration, and a
  different resource environment. → Report it as provenance only, not as an
  apples-to-apples performance baseline.
- [Risk] A 58-task Gantt chart can become crowded. → Use source-aware short
  labels only when a bar is wide enough and provide full labels in JSON.
- [Risk] Future WfFormat versions may change fields. → Fail with actionable
  validation errors instead of silently guessing.
