# HEFT Scheduling Under Uncertainty

This repository begins with a tested reproduction of the 10-task HEFT example
from:

H. Topcuoglu, S. Hariri, and M.-Y. Wu, "Performance-Effective and
Low-Complexity Task Scheduling for Heterogeneous Computing," *IEEE
Transactions on Parallel and Distributed Systems*, 13(3), 2002.
[DOI: 10.1109/71.993206](https://doi.org/10.1109/71.993206)

The long-term project studies how classical static scheduling behaves under
runtime uncertainty and dynamic workflow arrivals. It currently includes both
a deterministic paper benchmark and a trace-driven WfCommons baseline.

## Setup

The repository includes a local virtual environment created during development:

```bash
source .venv/bin/activate
python -m pip install -e ".[visualization,pegasus]"
```

Install the optional reinforcement-learning stack only for Phase 5:

```bash
python -m pip install -e ".[visualization,pegasus,rl]"
```

The `rl` extra installs Gymnasium, PyTorch, Stable-Baselines3, and
`sb3-contrib`. Non-RL commands do not import these packages.

The optional `pegasus` extra installs `pegasus-wms.api==5.1.2`, which provides
the Python classes used to define Pegasus workflows. The trace-driven HEFT
experiment does not require the full Pegasus planner or HTCondor because it
consumes an already published WfFormat execution trace.

Verify the API installation:

```bash
python -c "from Pegasus.api import Workflow; print(Workflow('check').name)"
```

## Phase 1: Paper Reproduction

- Paper Figure 3: 10 tasks, 15 communication edges, 3 processors.
- Mean computation costs and upward ranks.
- Non-increasing rank priority order.
- Insertion-based earliest-finish processor selection.
- Independent schedule validation.
- Published HEFT makespan of 80.
- Deterministic JSON output and optional Gantt chart.

Run:

```bash
heft-reproduce \
  --output results/paper_example_schedule.json \
  --plot results/paper_example_gantt.png
```

The command computes Table 1 ranks, the paper priority order, processor
assignments, schedule validity, and the published makespan of 80. Omit `--plot`
when Matplotlib is unavailable.

## Phase 2: WfCommons Trace Baseline

The project vendors one unchanged WfFormat 1.5 Montage execution trace:

- 58 tasks
- 114 dependency edges
- 111 files
- one observed Pegasus execution on Chameleon
- observed Pegasus makespan of 1060 seconds

Run the trace-driven simulation:

```bash
heft-wfcommons \
  --input data/raw/wfcommons/montage-chameleon-2mass-005d-001.json \
  --config configs/trace_worker_model.json \
  --output results/montage_trace_schedule.json \
  --plot results/montage_trace_gantt.png
```

The command parses and validates WfFormat, derives communication volumes from
shared files, creates a simulated three-worker runtime matrix, runs the same
HEFT scheduler, and reports:

- modeled makespan and schedule validity
- task count per worker
- per-worker and average utilization
- simulated speedup over the best serial worker
- cross-worker transfer bytes and communication time
- source-aware task IDs, programs, ranks, and assignments in JSON

The 1060-second Pegasus makespan is included only as trace provenance. It is not
directly comparable with the simulated HEFT makespan because the worker and
network models are different.

Worker assumptions and formulas are documented in
[`configs/README.md`](configs/README.md). Trace provenance and checksum are in
[`data/README.md`](data/README.md).

## Run Tests

The test suite uses Python's standard-library `unittest`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

To run without installing editable commands:

```bash
PYTHONPATH=src .venv/bin/python -m heft_reproduction.trace_cli
```

## Phase 3: Runtime Uncertainty

Phase 3 replays the already planned static HEFT assignment under uncertain
actual task durations. For a requested coefficient of variation (CV), each
task receives an independent mean-one log-normal multiplier. The task stays on
the worker chosen by the original HEFT plan, and each worker retains its planned
task order; actual delays still propagate through dependencies and
cross-worker communication.

Run 100 reproducible trials at four uncertainty levels:

```bash
heft-uncertainty \
  --cvs 0,0.1,0.3,0.5 \
  --trials 100 \
  --seed 20260719 \
  --output results/montage_uncertainty.json \
  --plot results/montage_uncertainty_summary.png
```

The report includes the fixed static-plan replay and a
`perfect_information_heft` reference. The latter reruns HEFT after observing
the sampled durations, so it is a diagnostic reference only: it is neither a
globally optimal oracle nor a deployable online scheduler. The difference is a
measure of how much the fixed plan loses to one specific form of perfect
runtime information, not proof that reinforcement learning will outperform
HEFT.

## Phase 4: Dynamic Multi-Workflow Scheduling

Phase 4 models complete workflow DAGs arriving over time. A workflow is hidden
before its arrival; after arrival, only tasks whose parents have completed and
whose inputs are available on a worker may start. Tasks are non-preemptive.
Runtime uncertainty uses the same mean-one log-normal model as Phase 3.

The dynamic simulator compares:

- `online-greedy-eft`: choose the currently feasible task-worker pair with the
  smallest estimated finish time.
- `per-workflow-static-heft`: keep each workflow's original HEFT assignment and
  within-workflow worker order while resolving cross-workflow contention.
- `rolling-heft`: reconsider unstarted tasks after every arrival, task
  completion, or data-availability event using estimated HEFT upward ranks.
- `aging-rolling-heft`: combine normalized upward rank with time since workflow
  arrival to reduce starvation of older workflows.
- `shortest-remaining-work`: prioritize the workflow with the least estimated
  unfinished work, then use normalized upward rank within that workflow.

Run the default reproducible comparison:

```bash
heft-dynamic \
  --input data/raw/wfcommons/montage-chameleon-2mass-005d-001.json \
  --config configs/trace_worker_model.json \
  --workflows 12 \
  --mean-interarrival 45 \
  --cv 0.3 \
  --seed 20260719 \
  --output results/dynamic_workflows.json \
  --plot results/dynamic_workflows_comparison.png
```

Repeat `--input PATH` to provide multiple WfFormat templates. Templates are
assigned to arriving instances round-robin. Inter-arrival times are sampled
from an exponential distribution with the requested mean; `0` makes all
workflows arrive at time zero.

The report includes per-workflow arrival, completion, response time, and JCT;
task assignments and queue wait; mean and P95 JCT/wait; throughput; worker
utilization; cross-worker transfers; decision count; candidate evaluations;
and measured scheduler wall time. Wall-clock timing is machine-dependent, while
the simulated schedule and candidate-evaluation count are reproducible.

The single-input command above remains useful for inspecting one dynamic
scenario. Phase 4B adds a controlled multi-family benchmark for evaluating
policy behavior across application structures and load conditions.

## Phase 4B: Multi-Family Dynamic Benchmark

The benchmark manifest contains unchanged WfFormat 1.5 traces from three
WfCommons families:

- Montage: 58-task benchmark trace and 310-task held-out scale trace.
- Epigenomics: 73-task benchmark trace and 445-task held-out scale trace.
- Seismology: 101-task benchmark trace and 201-task held-out scale trace.

Each sweep cell gives every policy the same workflow arrivals and sampled task
durations. Offered load is converted to a mean inter-arrival time by:

```text
mean_interarrival =
    mean_estimated_work_per_workflow / (worker_count * offered_load)
```

This is a normalized simulation input, not measured cluster utilization.
Realized utilization depends on DAG parallelism, communication, and policy
decisions.

Run the default small-trace benchmark:

```bash
heft-benchmark \
  --manifest configs/workflow_benchmark.json \
  --sizes small \
  --loads 0.6,1.0 \
  --cvs 0,0.3 \
  --seed-count 3 \
  --workflows 6 \
  --base-seed 20260719 \
  --output results/phase4b_benchmark.json \
  --plot results/phase4b_benchmark.png
```

Use `--sizes small,medium` to include the larger held-out scale traces. A
research result should use substantially more replications, such as
`--seed-count 30`; the default is deliberately small enough for local
iteration. The JSON records all scenario seeds, per-run metrics, sample means,
sample standard deviations, and 95% normal-approximation confidence intervals.

## Phase 5: Reinforcement-Learning Scheduling

Phase 5 keeps the validated event simulator and replaces only the policy
decision. The Gymnasium environment exposes estimated runtimes, HEFT-derived
rank, workflow age and remaining work, worker state, and a valid-action mask.
It never exposes a sampled actual duration before task completion.

For each simulated interval, the raw reward is:

```text
reward = -arrived_incomplete_workflows * elapsed_time
```

The undiscounted episode return therefore equals negative total workflow JCT.
Tests verify this identity and verify that all five heuristic adapters reproduce
their direct-simulator schedules.

Run the required toy learning gate:

```bash
heft-train-rl \
  --config configs/rl/toy_maskable_ppo.json \
  --model artifacts/rl/final_models/toy_maskable_ppo.zip \
  --output results/rl/toy_training.json \
  --plot results/rl/toy_learning_curve.png
```

The local seeded run improved held-out toy Mean JCT from `33.899` for a masked
random policy to `5.753`, matching Greedy-EFT and passing the predeclared 10%
gate.

Two WfCommons policy designs are retained:

- V1 candidate policy selects one of up to 128 feasible task-worker slots. Its
  flattened MLP improved only 7.6%-10.9% over random on small traces and did not
  generalize to medium traces.
- V2 hybrid policy selects among the five Phase 4B heuristic proposals. It
  reduced the V1 gap to the best heuristic from 23%-93% to approximately 0%-4%
  in the local held-out evaluation.

Train and evaluate the V2 policy:

```bash
heft-train-hybrid-rl \
  --config configs/rl/wfcommons_hybrid_maskable_ppo.json \
  --model artifacts/rl/final_models/wfcommons_hybrid_maskable_ppo.zip \
  --output results/rl/wfcommons_hybrid_training.json \
  --plot results/rl/wfcommons_hybrid_learning_curve.png

heft-evaluate-rl \
  --config configs/rl/wfcommons_hybrid_evaluation.json \
  --model artifacts/rl/final_models/wfcommons_hybrid_maskable_ppo.zip \
  --output results/rl/wfcommons_hybrid_evaluation.json \
  --plot results/rl/wfcommons_hybrid_evaluation.png
```

The V2 deterministic policy selected Greedy-EFT for more than 99% of held-out
decisions, with occasional Aging-HEFT choices. It slightly beat the best single
heuristic in one small setting and remained within about 4.2% in the other
small settings. This is evidence that the hybrid action abstraction is much
more learnable than V1, not evidence that RL generally outperforms the
heuristics. The local evaluation uses five seeds per small cell and one medium
scale seed; stronger conclusions require more traces and at least 30 held-out
seeds.

## Paper Data Provenance

`src/heft_reproduction/paper_example.py` manually transcribes:

- Figure 3's task graph and edge communication costs.
- Figure 3's computation-cost table for processors P1, P2, and P3.
- Table 1's published upward ranks.
- Section 4.2's published priority order.
- Figure 4(a)'s HEFT makespan of 80.

The fixture is intentionally kept separate from the algorithm. Tests compare
calculated results with the published values, making accidental edits visible.

## Interpretation

Phases 1 and 2 demonstrate static deterministic scheduling. Phase 3 evaluates
a fixed plan under controlled runtime uncertainty. Phase 4 adds online
multi-workflow contention and replanning baselines. Phase 4B evaluates those
baselines across three workflow families, two scales, controlled loads, and
repeated seeds. Phase 5 adds optional masked PPO policies and paired held-out
evaluation. All trace-driven phases use real workflow structure and
measurements, but worker heterogeneity, bandwidth, arrivals, and runtime
uncertainty remain explicit simulation assumptions. Results do not represent
measured GPU-cluster performance, establish statistical significance with only
a few seeds, or prove that HEFT or RL is optimal.

## Pegasus Boundary

Pegasus WMS and this simulator operate at different layers:

- `pegasus-wms.api`: defines jobs, files, transformations, and workflow DAGs.
- Full Pegasus planner plus HTCondor: plans and executes workflows on real
  resources; not installed or required for this phase.
- This project: parses an existing Pegasus-derived WfFormat trace and evaluates
  HEFT under documented simulated worker and network assumptions.

Installing the API does not turn `Compute-Fast`, `Balanced`, or `IO-Fast` into
measured hardware. They remain simulated workers.
