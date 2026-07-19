"""Small worker-affinity workflow used to validate that RL can learn."""

from __future__ import annotations

from ..dynamic_models import WorkflowTemplate
from ..dynamic_scenario import build_dynamic_scenario
from ..models import Workflow


TOY_PROCESSORS = ("Compute", "IO")


def build_toy_template() -> WorkflowTemplate:
    """Create two parallel chains with opposite worker affinities."""

    workflow = Workflow(
        tasks=(1, 2, 3, 4, 5, 6),
        processors=TOY_PROCESSORS,
        computation_costs={
            1: {"Compute": 1.0, "IO": 6.0},
            2: {"Compute": 6.0, "IO": 1.0},
            3: {"Compute": 1.0, "IO": 6.0},
            4: {"Compute": 6.0, "IO": 1.0},
            5: {"Compute": 1.0, "IO": 6.0},
            6: {"Compute": 6.0, "IO": 1.0},
        },
        communication_costs={
            (1, 3): 0.2,
            (3, 5): 0.2,
            (2, 4): 0.2,
            (4, 6): 0.2,
        },
    )
    return WorkflowTemplate(
        name="toy-worker-affinity",
        workflow=workflow,
        source_task_ids={task: f"toy-{task}" for task in workflow.tasks},
        programs={
            task: ("compute-task" if task % 2 else "io-task")
            for task in workflow.tasks
        },
        edge_data_bytes={
            edge: 200 for edge in workflow.communication_costs
        },
        source_filename="generated-toy.json",
    )


def build_toy_scenario(
    seed: int,
    workflow_count: int,
    mean_interarrival_time: float,
    runtime_cv: float,
):
    """Build a seeded toy episode using the production scenario generator."""

    return build_dynamic_scenario(
        templates=(build_toy_template(),),
        workflow_count=workflow_count,
        mean_interarrival_time=mean_interarrival_time,
        runtime_cv=runtime_cv,
        seed=seed,
    )

