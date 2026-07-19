# HEFT Scheduling Under Uncertainty

## Project Goal

Build a trace-driven research project that starts from a faithful reproduction of
the Heterogeneous Earliest Finish Time (HEFT) algorithm and later studies how
static scheduling degrades under runtime uncertainty and dynamic workflow
arrivals.

The project is intentionally staged. Each stage must produce a tested,
explainable result before the next stage begins.

## Research Story

1. Reproduce HEFT to understand classical static heterogeneous scheduling.
2. Measure sensitivity to inaccurate runtime estimates.
3. Add dynamically arriving workflows and rolling heuristic baselines.
4. Formulate online scheduling as a sequential decision problem.
5. Compare a learning-based scheduler with strong heuristic baselines.
6. Study when learning helps, when it fails, and how constraints affect it.

## Phase 1: Paper Reproduction

### Objective

Reproduce the 10-task example in Topcuoglu, Hariri, and Wu (2002) using the
paper's DAG, communication costs, and processor-specific computation costs.

### Deliverables

- A typed in-memory workflow model.
- A faithful fixture for Figure 3.
- Mean computation-cost and upward-rank calculations.
- Insertion-based HEFT processor selection.
- A deterministic command-line reproduction report.
- JSON and PNG schedule artifacts.
- Tests for ranks, task order, schedule validity, and makespan.

### Acceptance Criteria

- The upward ranks match Table 1 within floating-point tolerance.
- The priority order is `1, 3, 4, 2, 5, 6, 9, 7, 8, 10`.
- The generated schedule has no processor overlap.
- Every dependency and inter-processor communication delay is respected.
- The final makespan is `80`.
- The test suite passes without network access.

## Phase 2: WfCommons Integration

### Status

Completed for the vendored Montage `2mass-005d` WfFormat 1.5 instance.

### Deliverables

- Install and verify the optional Pegasus WMS Python API for future workflow
  definition experiments, while keeping trace parsing independent of it.
- Parse and semantically validate specification tasks, execution tasks, files,
  machines, dependencies, and observed runtime metadata.
- Map 58 tasks and 114 dependencies into the existing HEFT workflow model.
- Derive communication volume from 111 trace files.
- Generate processor-specific runtimes using a serialized CPU/I/O proxy model.
- Preserve source task IDs and program names through scheduling and reporting.
- Export deterministic JSON, utilization and transfer metrics, and a Gantt
  chart.
- Keep the observed Pegasus makespan separate as provenance.

### Acceptance Criteria

- The raw data SHA-256 matches the documented upstream fixture.
- The parser rejects invalid IDs, files, dependencies, runtimes, and cycles.
- Every trace task is scheduled exactly once.
- The resulting schedule has no overlap or dependency violations.
- Phase 1 continues to reproduce makespan 80.
- The complete test suite runs offline.

## Phase 3: Runtime Uncertainty

### Status

Implemented as a fixed-plan replay experiment with a perfect-information HEFT
diagnostic reference. Online heuristic baselines remain part of the next
dynamic-workflow phase.

- Treat historical/trace-derived runtimes as estimates.
- Sample task-level actual runtimes from mean-one log-normal distributions.
- Replay static HEFT assignments and per-worker task order under actual
  durations, including dependency and communication delays.
- Compare the fixed plan with HEFT rerun after seeing realized durations.
- Report per-trial values, means, sample standard deviations, and 95% normal
  approximation confidence intervals across seeded trials.

## Phase 4: Dynamic Workflows

### Status

Implemented as a non-preemptive event-driven simulator with complete DAG
arrivals, runtime uncertainty, and five transparent online baselines.

- Add seeded exponential workflow arrivals and multiple-template support.
- Process workflow arrival, task completion, and data-availability events.
- Preserve running tasks while allowing unstarted tasks to be reconsidered.
- Compare online greedy EFT, per-workflow static HEFT, rolling HEFT,
  aging-aware rolling HEFT, and shortest remaining work.
- Measure mean/P95 JCT, task wait, response time, throughput, utilization,
  communication, candidate evaluations, and scheduler wall time.
- Validate every schedule for coverage, arrival, worker overlap, dependency,
  communication, and realized-duration constraints.

## Phase 4B: Multi-Family Benchmark

### Status

Implemented with an offline, checksum-validated WfCommons corpus and a seeded
benchmark runner.

- Use Montage, Epigenomics, and Seismology traces to cover different DAG
  structures.
- Keep small traces for routine sweeps and larger traces as a held-out scale
  check.
- Convert normalized offered load into a documented mean inter-arrival time.
- Reuse one immutable scenario across all policies in each load/CV/seed cell.
- Report per-run metrics plus mean, sample standard deviation, and 95% normal
  confidence intervals.
- Export machine-readable JSON and a mean-JCT comparison plot.

This phase supports fair baseline evaluation, but a small local run is not a
statistical claim. Research conclusions should use more seeds, held-out
settings, and effect-size reporting.

## Phase 5: Learning-Based Scheduling

### Status

Implemented with a pausable simulator core, optional Gymnasium/PyTorch
dependencies, MaskablePPO training, and frozen held-out evaluation.

- Preserve exact Phase 4B heuristic schedules through the new decision core.
- Expose non-oracle observations and masked valid actions.
- Use negative active-workflow time as a dense reward exactly aligned with
  total JCT.
- Pass a seeded toy learning gate against masked random scheduling.
- Retain a failed-to-generalize low-level candidate policy as V1 evidence.
- Add a V2 hybrid policy that selects among five heuristic proposals.
- Evaluate V1, V2, random, and all five heuristics on shared small and medium
  WfCommons scenarios.

V2 closes most of V1's generalization gap but selects Greedy-EFT for more than
99% of held-out decisions. It is therefore a strong hybrid baseline and a
useful negative/diagnostic result, not a demonstrated replacement for the
heuristics.

## Research Guardrails

- Do not claim real GPU performance without measured GPU data.
- Do not claim RL outperforms HEFT before fair held-out evaluation.
- Give all schedulers the same observable information.
- Keep oracle results separate from deployable baselines.
- Record random seeds and configuration for every result.
- Treat negative results as findings, not failures to hide.

## Current Scope

Phases 1 through 5 are implemented. The next research milestone is broader
data and statistical evaluation: more workflow instances and families, at
least 30 paired test seeds, and a permutation-invariant candidate or graph
policy if low-level task-worker learning is revisited.
