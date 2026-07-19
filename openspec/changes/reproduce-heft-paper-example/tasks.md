## 1. Project Setup

- [x] 1.1 Create the Python package, CLI entry point, result directories, and project metadata
- [x] 1.2 Document installation, reproduction commands, and paper-data provenance

## 2. Workflow Model and Paper Fixture

- [x] 2.1 Implement typed workflow and schedule data models with structural validation
- [x] 2.2 Encode the paper's 10-task DAG, communication costs, computation matrix, expected ranks, and priority order

## 3. HEFT Implementation

- [x] 3.1 Implement mean computation costs, upward ranks, and deterministic task prioritization
- [x] 3.2 Implement dependency-ready times and insertion-based earliest-finish processor selection
- [x] 3.3 Implement independent schedule validation and makespan calculation

## 4. Reproduction Interface

- [x] 4.1 Implement the CLI report and deterministic JSON result export
- [x] 4.2 Implement an optional Matplotlib Gantt-chart generator

## 5. Verification

- [x] 5.1 Add tests for the paper fixture, ranks, priority order, processor assignments, and makespan 80
- [x] 5.2 Add tests for insertion behavior and invalid schedule detection
- [x] 5.3 Run the full test suite, generate final artifacts, visually inspect the chart, and validate the OpenSpec change
