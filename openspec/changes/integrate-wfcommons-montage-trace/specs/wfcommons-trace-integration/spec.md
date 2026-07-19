## ADDED Requirements

### Requirement: Immutable trace provenance
The system SHALL include an unmodified public WfCommons Montage instance and
SHALL document its source URL, upstream Git blob, SHA-256 checksum, schema
version, and license.

#### Scenario: Verify the vendored trace
- **WHEN** the documented checksum is calculated for the local raw JSON
- **THEN** it matches the recorded SHA-256 value

### Requirement: WfFormat trace parsing
The system SHALL parse WfFormat 1.5 specification tasks, files, execution tasks,
machines, and observed makespan into typed local records.

#### Scenario: Parse the Montage fixture
- **WHEN** the vendored `2mass-005d` instance is loaded
- **THEN** the parser reports 58 tasks, 114 dependency edges, 111 files, and one machine

### Requirement: Optional Pegasus workflow API
The system SHALL provide a pinned optional dependency that installs the Pegasus
workflow-definition Python API without requiring it for trace parsing or HEFT
scheduling.

#### Scenario: Verify the Pegasus API
- **WHEN** the project is installed with the `pegasus` extra
- **THEN** `Workflow`, `Job`, `Transformation`, and `File` import from `Pegasus.api`

### Requirement: Trace semantic validation
The system SHALL reject duplicate IDs, mismatched specification and execution
task sets, unknown task or file references, inconsistent parent/child
relationships, non-positive runtimes, and cyclic dependencies.

#### Scenario: Reject an unknown parent
- **WHEN** a task references a parent ID absent from the specification
- **THEN** parsing fails with an actionable validation error

### Requirement: Deterministic heterogeneous worker modeling
The system SHALL convert observed runtimes into a complete processor-specific
cost matrix using a serialized worker configuration and deterministic formula.

#### Scenario: Preserve the balanced-worker baseline
- **WHEN** a task is modeled on a worker with compute and I/O speed equal to one
- **THEN** its modeled runtime equals its observed runtime

#### Scenario: Prefer different workers for different CPU weights
- **WHEN** otherwise identical high-CPU and low-CPU tasks are modeled
- **THEN** the compute-fast and I/O-fast workers respectively produce the lower modeled runtimes

### Requirement: File-derived communication costs
The system SHALL calculate every dependency's data volume from shared parent
output and child input files and divide that volume by configured network
bandwidth to obtain an inter-worker communication cost.

#### Scenario: Convert shared bytes into seconds
- **WHEN** an edge shares 100,000,000 bytes and bandwidth is 100,000,000 bytes per second
- **THEN** its modeled communication cost is one second

### Requirement: Source identity preservation
The system SHALL retain a deterministic mapping between internal integer task
IDs and WfFormat source IDs, program names, runtimes, and CPU metrics.

#### Scenario: Report a scheduled source task
- **WHEN** a modeled task is included in JSON output
- **THEN** the entry contains both its internal task ID and original WfFormat task ID

### Requirement: Trace-driven HEFT report
The system SHALL run the existing HEFT scheduler on the modeled trace and
export deterministic JSON containing provenance, model configuration, trace
summary, schedule, validation status, makespan, processor utilization,
cross-worker communication bytes, and a simulated serial baseline.

#### Scenario: Schedule the Montage fixture
- **WHEN** the trace command runs with the default worker configuration
- **THEN** all 58 tasks are scheduled once, the schedule validates, and deterministic JSON is created

### Requirement: Trace schedule visualization
The system SHALL optionally create a processor-timeline PNG from the canonical
trace schedule with source-aware task labels.

#### Scenario: Generate the Montage Gantt chart
- **WHEN** plotting is requested and Matplotlib is installed
- **THEN** a non-empty PNG containing all configured processor timelines is created

### Requirement: Paper reproduction compatibility
The system SHALL preserve all Phase 1 paper ranks, assignments, validation
behavior, and makespan.

#### Scenario: Run the complete test suite
- **WHEN** Phase 1 and Phase 2 tests execute together
- **THEN** the paper fixture still reproduces a makespan of 80
