## ADDED Requirements

### Requirement: Deterministic uncertainty sampling

The system SHALL generate task-level log-normal duration multipliers from a
coefficient of variation and explicit random seed. The multipliers SHALL have
mean one in distribution, and a zero CV SHALL produce multipliers of exactly
one.

#### Scenario: Zero uncertainty preserves estimated durations

- **WHEN** an experiment is run with CV equal to zero
- **THEN** every actual task duration SHALL equal its estimated duration
- **AND** the fixed-plan replay makespan SHALL equal the original plan makespan

### Requirement: Fixed-plan execution replay

The system SHALL replay a static schedule using actual durations while
preserving each task's assigned worker and planned per-worker order. A replayed
task SHALL wait for both its actual dependency availability and the actual
completion of its preceding task on that worker.

#### Scenario: A late parent delays a child

- **WHEN** an actual parent duration is longer than estimated
- **THEN** a dependent child SHALL not start before the parent's actual finish
- **AND** required cross-worker communication time SHALL be respected

### Requirement: Uncertainty experiment report

The system SHALL evaluate one or more CV values over a configured number of
seeded trials and export per-trial and aggregate makespan results. The report
SHALL distinguish the deployable fixed-plan replay from the
perfect-information HEFT reference.

#### Scenario: Repeated seeded run is reproducible

- **WHEN** the same workflow, CV values, trial count, and seed are supplied
- **THEN** the report SHALL be byte-for-byte deterministic
