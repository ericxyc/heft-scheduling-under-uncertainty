# Trace Worker Model

`trace_worker_model.json` defines simulated heterogeneous workers. These values
are research assumptions, not measurements of the original Chameleon machine
and not claims about real CPU or GPU hardware.

For task `i` and worker `p`:

```text
cpu_weight_i = clamp(avgCPU_i / 100, 0, 1)

runtime_i,p = observed_runtime_i * (
    cpu_weight_i / compute_speed_p
    + (1 - cpu_weight_i) / io_speed_p
)
```

The `Balanced` worker reproduces the observed task runtime. `Compute-Fast`
favors tasks with high observed CPU utilization; `IO-Fast` favors tasks with
low observed CPU utilization.

For a dependency from task `i` to task `j`:

```text
shared_bytes_i,j = sum(
    size(file)
    for file in output_files_i intersect input_files_j
)

communication_seconds_i,j =
    shared_bytes_i,j / network_bandwidth_bytes_per_second
```

Communication is zero when both tasks are placed on the same worker, as defined
by the HEFT execution model.

## Benchmark Corpus Manifest

`workflow_benchmark.json` is the Phase 4B experiment manifest. It records each
trace's family, scale role, local path, official raw URL, SHA-256 checksum, and
expected task, edge, and file counts.

The benchmark loader verifies every checksum and count before constructing a
scenario. Entries tagged `small` are intended for routine policy sweeps;
entries tagged `medium` provide a larger held-out scale check. Both sizes use
the same transparent worker and communication assumptions described above.

## Reinforcement-Learning Configurations

RL configurations live under `configs/rl/`:

- `toy_maskable_ppo.json` defines the small worker-affinity learning gate.
- `wfcommons_maskable_ppo.json` defines the V1 low-level candidate policy.
- `wfcommons_evaluation.json` evaluates the V1 candidate policy.
- `wfcommons_hybrid_maskable_ppo.json` defines the V2 heuristic-selection
  policy.
- `wfcommons_hybrid_evaluation.json` evaluates V2 on the same frozen seeds as
  V1.

Training and evaluation seeds are disjoint. Training randomly selects from the
listed load and CV values. Evaluation uses one immutable scenario shared by
random, all five heuristics, and the frozen RL policy in each cell.
