## ADDED Requirements

### Requirement: Seeded dynamic workflow scenario

The system SHALL create one or more complete workflow instances with explicit
arrival times and task-level realized durations. Fixed templates, workflow
count, inter-arrival setting, CV, and seed SHALL reproduce the same scenario.

#### Scenario: Future workflow remains invisible

- **WHEN** a workflow's arrival time is greater than the current simulation time
- **THEN** none of its tasks SHALL be eligible for scheduling

### Requirement: Event-consistent non-preemptive execution

The system SHALL process workflow arrivals, task completions, and data
availability over a monotonic event clock. A started task SHALL retain its
worker until its realized duration completes.

#### Scenario: Cross-worker child waits for data

- **WHEN** a child is assigned to a different worker from a completed parent
- **THEN** the child SHALL not start before the parent's actual finish plus the
  modeled communication time

### Requirement: Comparable online scheduling policies

The system SHALL evaluate online greedy EFT, per-workflow static HEFT, and
rolling HEFT on the same immutable scenario. Policy choices SHALL use estimated
durations and SHALL not observe future arrivals or realized durations of
uncompleted tasks.

#### Scenario: Rolling policy responds to an event

- **WHEN** a task completes or a workflow arrives
- **THEN** rolling HEFT SHALL reconsider unstarted tasks
- **AND** it SHALL leave running tasks unchanged

### Requirement: Dynamic scheduling report

The system SHALL report per-workflow arrival/completion/JCT and aggregate mean
JCT, P95 JCT, task queue wait, makespan, throughput, worker utilization,
cross-worker transfer, decision count, candidate evaluations, and scheduler
wall time. It SHALL export task-level assignments and validation results.

#### Scenario: Complete valid simulation

- **WHEN** a scenario finishes
- **THEN** every task SHALL have executed exactly once
- **AND** no worker intervals SHALL overlap
- **AND** all arrival, dependency, and communication constraints SHALL hold
