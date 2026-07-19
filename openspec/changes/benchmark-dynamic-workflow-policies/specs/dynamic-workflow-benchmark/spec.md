## ADDED Requirements

### Requirement: Verified multi-family workflow corpus

The system SHALL load workflow templates from a manifest containing family,
size, source URL, local path, shape metadata, and SHA-256. It SHALL reject a
missing, modified, malformed, or shape-mismatched trace.

#### Scenario: Trace checksum mismatch

- **WHEN** a local trace does not match its manifest SHA-256
- **THEN** benchmark loading SHALL fail before scheduling

### Requirement: Completion-aware online baselines

The system SHALL provide aging rolling HEFT and shortest-remaining-work
policies using only arrived workflows, estimated runtimes, and observed system
state. They SHALL not observe future arrivals or uncompleted actual durations.

#### Scenario: Workflow age affects priority

- **WHEN** two feasible tasks have similar normalized ranks but one workflow has
  waited longer
- **THEN** aging rolling HEFT SHALL increase the older workflow's score

### Requirement: Fair benchmark sweep

The system SHALL create one immutable scenario per size, load, CV, and replicate
and evaluate every selected policy on that same scenario. Scenario seeds SHALL
be deterministic from a master seed.

#### Scenario: Policy comparison uses identical uncertainty

- **WHEN** multiple policies run for one sweep cell
- **THEN** they SHALL receive identical arrivals and actual task durations

### Requirement: Aggregate benchmark report

The system SHALL report per-run validity and core dynamic metrics and aggregate
mean, sample standard deviation, and 95% confidence interval by size, load, CV,
and policy. It SHALL export JSON and an optional comparison plot.

#### Scenario: Repeated seeded sweep

- **WHEN** a sweep is rerun with identical deterministic inputs
- **THEN** all simulated metrics and aggregate values excluding wall-clock
  scheduler timing SHALL match
