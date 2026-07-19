# Design: Phase 4B Benchmark

## Data

The manifest contains three workflow families and two sizes per family. Raw
WfFormat files are vendored unchanged and verified by SHA-256 before use.
Small traces form the default benchmark set; medium traces are held out for
scale and generalization checks.

## Added Policies

### Aging Rolling HEFT

For a ready task `i` in workflow `k`:

```text
normalized_rank = rank_u(i) / max_rank(k)
normalized_age = (now - arrival(k)) / static_HEFT_makespan(k)
score = normalized_rank + aging_weight * normalized_age
```

The highest score is selected, followed by the feasible worker with minimum
estimated finish time. Normalization makes ranks and ages comparable across
workflow families and scales.

### Shortest Remaining Work

For each arrived workflow, sum the mean estimated computation cost of its
unstarted tasks. Select a ready task from the workflow with the least remaining
work, use normalized upward rank within that workflow, then choose the feasible
worker with minimum estimated finish time. Running tasks are non-preemptive.

## Offered Load

For the round-robin template mix in one scenario:

```text
reference_work = mean(sum_i mean_worker_runtime(i))
mean_interarrival = reference_work / (worker_count * offered_load)
```

This is a transparent workload-normalization proxy. Communication and policy
placement can shift realized utilization, so offered load is not a guarantee
of exact utilization.

## Sweep

Each `(size, load, CV, replicate)` creates one immutable scenario shared by all
policies. Core run metrics and validity are retained without task-level
schedules to keep reports compact. Aggregates report sample mean, sample
standard deviation, and normal-approximation 95% confidence half-width.

Scheduler wall time is retained per run but treated as machine-dependent.
