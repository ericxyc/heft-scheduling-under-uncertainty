# Phase 5 and Phase 6 Artifacts

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

Phase 6 reports:

- `wfcommons_gnn_training.*`: direct task-worker GNN+PPO training diagnostics.
- `wfcommons_gnn_legacy_evaluation.*`: direct GNN legacy evaluation.
- `wfcommons_gnn_hybrid_training.*`: graph-hybrid PPO diagnostics.
- `wfcommons_gnn_hybrid_legacy_evaluation.*`: exact legacy-protocol comparison.
- `wfcommons_gnn_hybrid_key_evaluation.*`: five-seed medium held-out result.
- `wfcommons_gnn_hybrid_fair_training.*`: zero-prior, entropy-decay PPO
  training diagnostics and training action counts.
- `wfcommons_gnn_hybrid_fair_key_evaluation.*`: five-seed evaluation of the
  fair PPO baseline.

The Phase 6 graph-hybrid model robustly reproduces Greedy rather than beating
it. Model archives and periodic checkpoints remain excluded from version
control.

The fair baseline sampled all five heuristics during training but selected
shortest remaining work for every deterministic held-out decision. Its mean
JCT was `952.095`, 6.4% worse than Greedy. Removing the Greedy prior fixed the
exploration imbalance but did not produce graph-conditioned switching.
