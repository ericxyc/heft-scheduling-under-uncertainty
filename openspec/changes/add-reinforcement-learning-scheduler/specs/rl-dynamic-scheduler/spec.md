## ADDED Requirements

### Requirement: Pausable simulator core

The system SHALL expose dynamic scheduling as a deterministic sequence of event
processing, candidate selection, task commitment, and clock advancement. The
existing heuristic entry point SHALL continue to produce the same logical
schedule and metrics for fixed scenarios.

#### Scenario: Existing heuristic uses the decision core

- **WHEN** an existing policy schedules a seeded scenario through the refactored
  simulator
- **THEN** task placements, simulated times, event counts, and logical metrics
  SHALL match the pre-refactor result

### Requirement: Valid masked RL environment

The system SHALL provide a Gymnasium-compatible environment with fixed declared
observation and action spaces. Every valid action SHALL represent one currently
feasible task-worker pair, and invalid actions SHALL be masked and rejected.

#### Scenario: RL selects a valid candidate

- **WHEN** the agent selects a slot marked valid
- **THEN** exactly that task SHALL start non-preemptively on that idle worker

### Requirement: JCT-aligned reward

The system SHALL accumulate negative active-workflow time as its raw reward.

#### Scenario: Episode completes

- **WHEN** all workflows have completed
- **THEN** undiscounted raw episode return SHALL equal negative total workflow
  JCT within floating-point tolerance

### Requirement: No oracle observation

The environment SHALL expose estimated runtimes and observed completion state
but SHALL NOT expose sampled actual durations for uncompleted tasks.

#### Scenario: Runtime uncertainty is enabled

- **WHEN** a task has not completed
- **THEN** its observation SHALL be independent of its sampled actual multiplier

### Requirement: Optional reproducible RL training

The system SHALL keep RL dependencies optional and provide seeded MaskablePPO
training, checkpointing, and frozen-policy evaluation.

#### Scenario: RL extras are not installed

- **WHEN** a user runs a non-RL HEFT or dynamic command
- **THEN** the command SHALL work without importing PyTorch or Gymnasium

### Requirement: Held-out comparison

The system SHALL compare a frozen learned policy and all selected heuristics on
identical held-out scenarios and report validity and paired scheduling metrics.

#### Scenario: Evaluation scenario is shared

- **WHEN** RL and heuristic policies are evaluated for one test seed
- **THEN** they SHALL receive the same arrivals, actual runtimes, workers, and
  communication costs
