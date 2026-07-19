## ADDED Requirements

### Requirement: Paper fixture fidelity
The system SHALL provide the HEFT paper's 10-task DAG, 15 communication costs,
three-processor computation matrix, expected upward ranks, and expected task
priority order as a deterministic fixture.

#### Scenario: Load the published example
- **WHEN** the paper fixture is loaded
- **THEN** it contains tasks 1 through 10, processors P1 through P3, and the published computation and communication costs

### Requirement: Upward-rank calculation
The system SHALL calculate each task's upward rank from mean computation costs,
mean communication costs, and the maximum successor path.

#### Scenario: Reproduce Table 1 ranks
- **WHEN** upward ranks are calculated for the paper fixture
- **THEN** every rank matches the published value within a tolerance of 0.001

### Requirement: Deterministic task prioritization
The system SHALL sort tasks by non-increasing upward rank with deterministic
tie-breaking.

#### Scenario: Reproduce the published priority order
- **WHEN** the paper fixture tasks are prioritized
- **THEN** the order is 1, 3, 4, 2, 5, 6, 9, 7, 8, 10

### Requirement: Insertion-based processor selection
The system SHALL evaluate every processor and place each task in the earliest
feasible timeline gap that respects processor occupancy, dependencies, and
communication delays.

#### Scenario: Select the earliest finishing processor
- **WHEN** a prioritized task has multiple feasible processor placements
- **THEN** the placement with the smallest earliest finish time is committed

#### Scenario: Insert into an internal idle gap
- **WHEN** a processor has an internal idle gap large enough for a ready task
- **THEN** the candidate placement uses that gap instead of appending after the final scheduled task

### Requirement: Schedule validation
The system SHALL independently reject schedules with missing or duplicate
tasks, incorrect durations, processor overlap, precedence violations, or
communication-delay violations.

#### Scenario: Validate the reproduced schedule
- **WHEN** the generated paper schedule is validated
- **THEN** no violations are reported

#### Scenario: Reject an overlapping schedule
- **WHEN** two tasks overlap on the same processor
- **THEN** validation reports a processor-overlap violation

### Requirement: Published makespan reproduction
The system SHALL reproduce the paper example's HEFT makespan.

#### Scenario: Complete the paper example
- **WHEN** HEFT schedules the published fixture
- **THEN** the makespan is 80 within a tolerance of 0.001

### Requirement: Reproducible command-line report
The system SHALL provide a command that prints ranks and schedule entries and
writes the schedule, priority order, validation result, and makespan to JSON.

#### Scenario: Run the reproduction command
- **WHEN** the user runs the project reproduction command
- **THEN** the command exits successfully and creates a deterministic result JSON file

### Requirement: Human-readable visualization
The system SHALL provide an optional Gantt-chart generator based on the
canonical schedule result.

#### Scenario: Matplotlib is available
- **WHEN** visualization is requested and Matplotlib is installed
- **THEN** a PNG showing each processor timeline and task interval is created
