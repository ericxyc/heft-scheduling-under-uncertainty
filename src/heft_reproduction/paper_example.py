"""Reviewed transcription of the HEFT paper's Figure 3 example."""

from __future__ import annotations

from .models import ScheduleEntry, Workflow


PAPER_DOI = "10.1109/71.993206"
EXPECTED_MAKESPAN = 80.0
EXPECTED_PRIORITY_ORDER = (1, 3, 4, 2, 5, 6, 9, 7, 8, 10)

# Table 1 values are rounded to three decimals in the paper.
EXPECTED_UPWARD_RANKS = {
    1: 108.000,
    2: 77.000,
    3: 80.000,
    4: 80.000,
    5: 69.000,
    6: 63.333,
    7: 42.667,
    8: 35.667,
    9: 44.333,
    10: 14.667,
}

# Figure 4(a), reconstructed from the published processor timelines.
EXPECTED_SCHEDULE = {
    1: ScheduleEntry(1, "P3", 0.0, 9.0),
    2: ScheduleEntry(2, "P1", 27.0, 40.0),
    3: ScheduleEntry(3, "P3", 9.0, 28.0),
    4: ScheduleEntry(4, "P2", 18.0, 26.0),
    5: ScheduleEntry(5, "P3", 28.0, 38.0),
    6: ScheduleEntry(6, "P2", 26.0, 42.0),
    7: ScheduleEntry(7, "P3", 38.0, 49.0),
    8: ScheduleEntry(8, "P1", 57.0, 62.0),
    9: ScheduleEntry(9, "P2", 56.0, 68.0),
    10: ScheduleEntry(10, "P2", 73.0, 80.0),
}


def load_paper_example() -> Workflow:
    """Return the paper's 10-task, 3-processor HEFT example."""

    computation_costs = {
        1: {"P1": 14.0, "P2": 16.0, "P3": 9.0},
        2: {"P1": 13.0, "P2": 19.0, "P3": 18.0},
        3: {"P1": 11.0, "P2": 13.0, "P3": 19.0},
        4: {"P1": 13.0, "P2": 8.0, "P3": 17.0},
        5: {"P1": 12.0, "P2": 13.0, "P3": 10.0},
        6: {"P1": 13.0, "P2": 16.0, "P3": 9.0},
        7: {"P1": 7.0, "P2": 15.0, "P3": 11.0},
        8: {"P1": 5.0, "P2": 11.0, "P3": 14.0},
        9: {"P1": 18.0, "P2": 12.0, "P3": 20.0},
        10: {"P1": 21.0, "P2": 7.0, "P3": 16.0},
    }

    communication_costs = {
        (1, 2): 18.0,
        (1, 3): 12.0,
        (1, 4): 9.0,
        (1, 5): 11.0,
        (1, 6): 14.0,
        (2, 8): 19.0,
        (2, 9): 16.0,
        (3, 7): 23.0,
        (4, 8): 27.0,
        (4, 9): 23.0,
        (5, 9): 13.0,
        (6, 8): 15.0,
        (7, 10): 17.0,
        (8, 10): 11.0,
        (9, 10): 13.0,
    }

    return Workflow(
        tasks=tuple(range(1, 11)),
        processors=("P1", "P2", "P3"),
        computation_costs=computation_costs,
        communication_costs=communication_costs,
    )
