"""Build dynamic arrival scenarios from trace-derived workflow templates."""

from __future__ import annotations

from math import isfinite
from random import Random
from typing import Sequence

from .dynamic_models import (
    DynamicScenario,
    DynamicWorkflowInstance,
    WorkflowTemplate,
)
from .trace_model import ModeledTraceWorkflow
from .uncertainty import (
    sample_duration_multipliers,
    workflow_with_actual_durations,
)


def template_from_modeled(
    modeled: ModeledTraceWorkflow,
    name: str | None = None,
) -> WorkflowTemplate:
    """Preserve source labels and edge bytes in a dynamic template."""

    template_name = name or modeled.trace.source_path.stem
    return WorkflowTemplate(
        name=template_name,
        workflow=modeled.workflow,
        source_task_ids=dict(modeled.internal_to_source),
        programs={
            task: modeled.task_metadata[task].program
            for task in modeled.workflow.tasks
        },
        edge_data_bytes=dict(modeled.edge_data_bytes),
        source_filename=modeled.trace.source_path.name,
    )


def build_dynamic_scenario(
    templates: Sequence[WorkflowTemplate],
    workflow_count: int,
    mean_interarrival_time: float,
    runtime_cv: float,
    seed: int,
) -> DynamicScenario:
    """Create seeded arrivals and actual durations shared by all policies."""

    if not templates:
        raise ValueError("at least one workflow template is required")
    if workflow_count <= 0:
        raise ValueError("workflow count must be positive")
    if not isfinite(mean_interarrival_time) or mean_interarrival_time < 0:
        raise ValueError(
            "mean inter-arrival time must be finite and non-negative"
        )
    if not isfinite(runtime_cv) or runtime_cv < 0:
        raise ValueError("runtime CV must be finite and non-negative")
    if len({template.name for template in templates}) != len(templates):
        raise ValueError("workflow template names must be unique")

    processors = templates[0].workflow.processors
    if any(template.workflow.processors != processors for template in templates):
        raise ValueError("all workflow templates must use the same workers")

    generator = Random(seed)
    instances: list[DynamicWorkflowInstance] = []
    arrival_time = 0.0
    for index in range(workflow_count):
        if index and mean_interarrival_time > 0:
            arrival_time += generator.expovariate(
                1.0 / mean_interarrival_time
            )
        template = templates[index % len(templates)]
        runtime_seed = generator.randrange(0, 2**63)
        multipliers = sample_duration_multipliers(
            template.workflow.tasks,
            runtime_cv,
            runtime_seed,
        )
        instances.append(
            DynamicWorkflowInstance(
                id=f"W{index + 1:04d}",
                template=template,
                arrival_time=arrival_time,
                actual_workflow=workflow_with_actual_durations(
                    template.workflow,
                    multipliers,
                ),
                runtime_seed=runtime_seed,
            )
        )

    return DynamicScenario(
        instances=tuple(instances),
        runtime_cv=runtime_cv,
        mean_interarrival_time=mean_interarrival_time,
        master_seed=seed,
    )
