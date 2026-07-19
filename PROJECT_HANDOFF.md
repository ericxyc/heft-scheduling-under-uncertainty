# Project Handoff

## Goal

Build a credible undergraduate research project on dynamic scientific-workflow scheduling for outreach to Lei Ying's group. The research narrative is:

1. Reproduce the HEFT paper example.
2. Use WfCommons traces in a transparent trace-driven simulation.
3. Introduce runtime uncertainty and dynamic workflow arrivals.
4. Compare online scheduling heuristics.
5. Explore reinforcement learning honestly, without claiming it already outperforms strong heuristics.

## What Is Implemented

- HEFT paper-example reproduction, including ranks, task-worker assignments, makespan, validation, JSON output, and Gantt chart.
- WfCommons trace parsing for Montage, Epigenomics, and Seismology DAG instances.
- Trace-driven worker and communication simulation. Observed workflow runtime and file sizes are real trace inputs; heterogeneous workers and network settings are explicit simulation assumptions, not real GPU measurements.
- Runtime uncertainty experiments using coefficient of variation (CV).
- Dynamic multi-workflow event simulator and five baselines:
  - `online-greedy-eft`
  - `per-workflow-static-heft`
  - `rolling-heft`
  - `aging-rolling-heft`
  - `shortest-remaining-work`
- Two RL formulations using Maskable PPO:
  - V1 selects a task-worker candidate directly.
  - V2 selects one recommended next action from the five heuristic policies.

## Key Results

### Toy RL gate

The V1 direct task-worker policy learned the toy environment:

- Random mean JCT: `33.899`
- Learned-policy mean JCT: `5.753`
- Greedy mean JCT: `5.753`

Here, Random chooses uniformly among currently legal task-worker actions and ignores runtime, communication, worker availability, and task priority.

### WfCommons RL evaluation

The V1 direct policy did not generalize reliably to real WfCommons workflows. The V2 heuristic-selection policy learned a near-Greedy behavior with rare Aging corrections.

- Medium, load `0.8`, CV `0.2`: RL JCT `894.300`; best heuristic (Greedy) `889.080` (`-0.59%`).
- Small, load `0.6`, CV `0.3`: RL JCT `192.454`; best heuristic (Greedy) `193.651` (`+0.62%`).
- Other held-out cells lagged the best heuristic by about `1.4%` to `4.2%`.

Correct conclusion: the current RL policy reaches a reasonable near-Greedy policy but does not yet show broad, statistically credible superiority over the strongest heuristic baseline.

## How To Run

Create a fresh virtual environment on each machine; do not copy `.venv` between macOS, Linux, or Windows.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[rl]"
pytest -q
```

Useful commands:

```bash
heft-reproduce --output results/paper_example_schedule.json --plot results/paper_example_gantt.png
heft-train-rl --help
heft-train-wfcommons-rl --help
heft-train-hybrid-rl --help
heft-evaluate-rl --help
```

Training configurations are in `configs/rl/`. The generated model checkpoints under `artifacts/rl/final_models/` are intentionally ignored by Git.

## Recommended Next Phase

Strengthen the direct RL approach before making any claim that RL is better than heuristics:

1. Expand to roughly 60-100 DAG instances across at least 3-5 WfCommons workflow families.
2. Randomize arrivals, runtime uncertainty, and worker profiles with 10-20 training seeds per instance.
3. Split train/validation/test by DAG instance, not merely by random seed.
4. Replace the flat candidate representation with a DAG-aware GNN state encoder.
5. Evaluate on 20-30 fully held-out DAGs with 30 paired seeds per scenario.

Start with a 10k-episode smoke test and validate the learning curve and held-out results before committing to a long training run.

## Intel Laptop Workflow

Clone this repository on the Intel machine, create a new environment, and install dependencies. For Windows, WSL2 is recommended. If it has an NVIDIA GPU, install the matching CUDA build of PyTorch and expose a configurable `cuda` device in the training script; the scheduling simulator itself remains CPU-heavy.

## Outreach Framing

Avoid claiming deployment or real-GPU benchmark results. A precise summary for research outreach is:

> I reproduced HEFT and built a trace-driven dynamic scheduling simulator. A direct task-worker RL policy did not generalize reliably to real DAGs, so I reformulated the action space as selecting among online heuristic proposals. The hybrid policy converged to near-Greedy behavior with occasional aging corrections, which motivated my interest in richer DAG-aware representations and evaluation under uncertainty.
