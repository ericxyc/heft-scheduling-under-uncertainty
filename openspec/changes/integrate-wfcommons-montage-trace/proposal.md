## Why

The paper fixture proves the HEFT implementation is correct, but it is too small
and synthetic to support a credible study of scheduling under uncertainty.
Importing a public Pegasus execution trace provides a realistic DAG, observed
runtimes, and file-derived communication volumes while keeping every simulation
assumption explicit and reproducible.

## What Changes

- Vendor one unmodified WfCommons Montage WfFormat 1.5 instance with source,
  license, Git blob, and SHA-256 provenance.
- Provide a pinned optional Pegasus WMS Python API installation for future
  workflow-definition experiments without making it a trace-parser dependency.
- Parse workflow specification and execution records using the Python standard
  library, with structural and semantic validation.
- Convert trace tasks into the existing HEFT `Workflow` model using stable
  integer IDs, a documented three-worker simulation profile, and communication
  costs derived from shared file sizes and configured bandwidth.
- Add a trace-specific CLI that prints dataset summaries and scheduling metrics,
  exports deterministic JSON, and optionally creates a labeled Gantt chart.
- Preserve the paper reproduction behavior and avoid treating simulated worker
  costs as measured heterogeneous hardware performance.

## Capabilities

### New Capabilities

- `wfcommons-trace-integration`: Deterministic import, modeling, scheduling, and
  reporting for a WfFormat workflow execution trace.

### Modified Capabilities

None.

## Impact

- Adds a raw data fixture and provenance documentation under `data/`.
- Adds an optional `pegasus` dependency extra plus parser, simulation-model,
  metrics, CLI, and visualization modules under
  `src/heft_reproduction/`.
- Adds a `heft-wfcommons` console entry point and Phase 2 tests.
- Reuses the existing `Workflow`, `schedule_heft`, and validation APIs without
  changing the paper example or its expected results.
