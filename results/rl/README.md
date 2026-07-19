# Phase 5 Artifacts

Generated Phase 5 reports:

- `toy_training.json` and `toy_learning_curve.png`: required toy learning gate.
- `wfcommons_training.json` and `wfcommons_learning_curve.png`: V1 candidate
  policy training.
- `wfcommons_evaluation.json` and `wfcommons_evaluation.png`: frozen V1
  held-out comparison.
- `wfcommons_hybrid_training.json` and
  `wfcommons_hybrid_learning_curve.png`: V2 hybrid training.
- `wfcommons_hybrid_evaluation.json` and
  `wfcommons_hybrid_evaluation.png`: frozen V2 held-out comparison.

Model archives are generated under `artifacts/rl/final_models/` and excluded
from version control. JSON reports contain the exact configuration, scenario
seeds, per-run metrics, validity, confidence intervals, and V2 heuristic action
counts.
