"""DAG-aware observations for graph scheduling policies."""

from __future__ import annotations

import numpy as np

from ..dynamic_simulator import COMPLETED, PENDING, RUNNING, DynamicSchedulingCore
from .observation import EncodedObservation, _scales


GRAPH_NODE_FEATURES = 14
GRAPH_EDGE_FEATURES = 1


def _node_row(
    core: DynamicSchedulingCore,
    workflow_id: str,
    task: int,
    ready_refs: set[tuple[str, int]],
    time_scale: float,
    work_scale: float,
    max_tasks: int,
) -> np.ndarray:
    instance = core.instances[workflow_id]
    workflow = instance.template.workflow
    status = core.status[instance.ref(task)]
    costs = [workflow.computation_cost(task, p) for p in workflow.processors]
    incoming = sum(
        workflow.communication_cost(parent, task)
        for parent in workflow.predecessors(task)
    )
    outgoing = sum(
        workflow.communication_cost(task, child)
        for child in workflow.successors(task)
    )
    remaining = sum(
        workflow.mean_computation_cost(other)
        for other in workflow.tasks
        if core.status[instance.ref(other)] != COMPLETED
    )
    maximum_rank = max(core.ranks[workflow_id].values())
    age = max(0.0, core.now - instance.arrival_time)
    row = np.asarray(
        [
            float(status == PENDING),
            float(status == RUNNING),
            float(status == COMPLETED),
            core.ranks[workflow_id][task] / max(maximum_rank, 1e-9),
            workflow.mean_computation_cost(task) / time_scale,
            min(costs) / time_scale,
            max(costs) / time_scale,
            len(workflow.predecessors(task)) / max_tasks,
            len(workflow.successors(task)) / max_tasks,
            incoming / time_scale,
            outgoing / time_scale,
            age / max(core.baseline_makespans[workflow_id], 1e-9),
            float(instance.ref(task) in ready_refs),
            remaining / work_scale,
        ],
        dtype=np.float32,
    )
    return np.clip(row, 0.0, 10.0)


def encode_graph_observation(
    core: DynamicSchedulingCore,
    base: EncodedObservation,
    max_nodes: int,
    max_edges: int,
) -> EncodedObservation:
    """Augment candidate features with the arrived workflow DAGs."""

    arrived = [
        instance
        for instance in core.scenario.instances
        if instance.id in core.arrived_ids
    ]
    node_refs = [
        instance.ref(task)
        for instance in arrived
        for task in instance.template.workflow.topological_order()
    ]
    if len(node_refs) > max_nodes:
        raise ValueError(
            f"graph has {len(node_refs)} arrived tasks, exceeds {max_nodes}"
        )
    node_index = {ref: index for index, ref in enumerate(node_refs)}
    edge_refs = [
        (instance.ref(parent), instance.ref(child))
        for instance in arrived
        for parent, child in instance.template.workflow.communication_costs
    ]
    edge_refs = edge_refs[:max_edges]

    nodes = np.zeros((max_nodes, GRAPH_NODE_FEATURES), dtype=np.float32)
    node_mask = np.zeros(max_nodes, dtype=np.int8)
    edge_index = np.zeros((max_edges, 2), dtype=np.int32)
    edge_features = np.zeros(
        (max_edges, GRAPH_EDGE_FEATURES), dtype=np.float32
    )
    edge_mask = np.zeros(max_edges, dtype=np.int8)
    candidate_nodes = np.zeros(base.values["action_mask"].shape, dtype=np.int32)
    time_scale, work_scale, max_tasks = _scales(core)
    ready_refs = {candidate.ref for candidate in base.slots}

    for index, (workflow_id, task) in enumerate(node_refs):
        nodes[index] = _node_row(
            core,
            workflow_id,
            task,
            ready_refs,
            time_scale,
            work_scale,
            max_tasks,
        )
        node_mask[index] = 1
    for index, (source, target) in enumerate(edge_refs):
        edge_index[index] = (node_index[source], node_index[target])
        instance = core.instances[source[0]]
        communication = instance.template.workflow.communication_cost(
            source[1], target[1]
        )
        edge_features[index, 0] = min(10.0, communication / time_scale)
        edge_mask[index] = 1
    for index, candidate in enumerate(base.slots):
        candidate_nodes[index] = node_index[candidate.ref]

    values = dict(base.values)
    values.update(
        {
            "graph_nodes": nodes,
            "graph_node_mask": node_mask,
            "graph_edge_index": edge_index,
            "graph_edge_features": edge_features,
            "graph_edge_mask": edge_mask,
            "candidate_nodes": candidate_nodes,
        }
    )
    return EncodedObservation(
        values=values,
        slots=base.slots,
        raw_candidate_count=base.raw_candidate_count,
        truncated_count=base.truncated_count,
    )
