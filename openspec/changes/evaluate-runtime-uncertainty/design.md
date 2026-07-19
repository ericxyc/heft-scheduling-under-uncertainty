# Design: Runtime-Uncertainty Evaluation

## Experimental Model

The static HEFT plan is created from the modeled trace workflow and remains
fixed during replay: each task retains its assigned worker and each worker
retains the order induced by the planned schedule. Actual task durations are
sampled as

```text
actual_duration(i, p) = estimated_duration(i, p) * Z_i
```

where `Z_i` is independent log-normal noise with mean one and requested
coefficient of variation (CV). A task-level multiplier preserves the modeled
relative affinity between workers while making the estimate inaccurate.

For `CV > 0`, the log-normal parameters are:

```text
sigma = sqrt(log(1 + CV^2))
mu = -sigma^2 / 2
Z_i ~ LogNormal(mu, sigma)
```

For `CV = 0`, every multiplier is exactly one.

## Static-Plan Replay

The replay is event-consistent rather than a cosmetic shift of Gantt bars. A
task can start only after:

1. all parents have actually completed and any cross-worker transfer completes;
2. the previous task in its fixed worker order has actually completed.

This permits early starts when tasks finish early and propagates delays when
they finish late, without granting the static policy any new decisions.

## Reference

For each sampled realization, HEFT is also rerun on the realized duration
matrix. This is labelled `perfect_information_heft`: it has advance knowledge
of the sampled durations and is therefore a diagnostic reference, not a true
optimal oracle and not a fair online deployment baseline.

## Outputs

The JSON report records the baseline planned makespan, CV level, trial seed,
static replay makespan, perfect-information HEFT makespan, and aggregate mean,
sample standard deviation, and 95% normal-approximation confidence interval.
An optional plot shows mean makespan and the static-plan gap across CV levels.
