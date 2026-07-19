## Why

The research project needs a trustworthy implementation of classical HEFT
scheduling before uncertainty, dynamic arrivals, or reinforcement learning can
be evaluated. Reproducing the paper's published 10-task example provides a
small, deterministic correctness benchmark for every later experiment.

## What Changes

- Add a standalone Python package for representing heterogeneous DAG workflows.
- Encode the HEFT paper's Figure 3 task graph and computation-cost matrix as a
  reviewed fixture.
- Implement mean computation costs, upward ranks, insertion-based processor
  selection, and makespan calculation.
- Add schedule validation for processor overlap, precedence, and communication
  constraints.
- Add a CLI that prints the reproduction results and writes machine-readable
  schedule output.
- Add automated tests against the paper's task order, rank values, and
  makespan of 80.
- Add a schedule visualization for human inspection.

## Capabilities

### New Capabilities

- `heft-paper-reproduction`: Reproduce and validate the HEFT paper's 10-task
  static scheduling example.

### Modified Capabilities

None.

## Impact

This creates a new self-contained Python project and does not modify the
existing resume-generation or job-search code. The implementation uses the
Python standard library for scheduling and tests; Matplotlib is used only for
the optional PNG visualization.
