## 1. Data and Configuration

- [x] 1.1 Vendor the official Montage WfFormat JSON and document URL, Git blob, checksum, schema, and license
- [x] 1.2 Add a serialized default worker and network configuration with documented formulas and limitations
- [x] 1.3 Install, pin, verify, and document the optional Pegasus WMS Python API

## 2. WfFormat Parsing

- [x] 2.1 Implement typed trace, task, execution, file, and machine records
- [x] 2.2 Implement JSON loading and semantic validation for IDs, references, runtimes, dependencies, and DAG structure

## 3. Trace Modeling

- [x] 3.1 Implement stable task-ID mapping and CPU/I/O proxy computation-cost generation
- [x] 3.2 Implement shared-file byte accounting and bandwidth-based communication costs
- [x] 3.3 Preserve source task metadata and model assumptions in a modeled-workflow result

## 4. Reporting

- [x] 4.1 Implement schedule metrics for utilization, serial baseline, speedup, and cross-worker communication
- [x] 4.2 Implement the `heft-wfcommons` CLI and deterministic JSON export
- [x] 4.3 Implement a source-aware trace Gantt chart

## 5. Verification and Documentation

- [x] 5.1 Add parser and semantic-validation tests with focused synthetic fixtures
- [x] 5.2 Add model-formula, communication, metrics, CLI, and vendored Montage integration tests
- [x] 5.3 Update project documentation and Phase 2 status without changing Phase 1 usage
- [x] 5.4 Run all tests, generate and inspect final artifacts, verify the raw checksum, and strictly validate OpenSpec
