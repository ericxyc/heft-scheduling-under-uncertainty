# Benchmark Dynamic Workflow Policies

## Why

Phase 4 validates online scheduling with repeated copies of one small Montage
trace. It does not test cross-application structure, scale, repeated random
scenarios, or strong completion-aware heuristics. These gaps must be addressed
before an RL policy can be evaluated credibly.

## What Changes

- Vendor and checksum six WfCommons traces covering Montage, Epigenomics, and
  Seismology at small and medium scales.
- Add a benchmark manifest with family, size, source, shape, and split metadata.
- Add aging rolling HEFT and non-preemptive shortest-remaining-work baselines.
- Add a seeded benchmark sweep across workflow size, offered load, runtime CV,
  policy, and replicate.
- Aggregate mean, standard deviation, and 95% confidence intervals and produce
  comparison plots.

## Non-goals

- This change does not train reinforcement learning policies.
- Medium traces are held-out scalability inputs; the default local sweep uses
  small traces to keep runtime practical.
- Simulated heterogeneous workers are still explicit modeling assumptions, not
  measured GPU hardware.
