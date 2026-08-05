"""Classical greedy repair shared by the comparison baselines."""

from __future__ import annotations

import random

from ..distance import DistanceMatrix
from ..greedy import insertion_cost
from ..models import TOPInstance, TOPSolution
from ..solution import (
    calculate_route_distance,
    update_solution_metrics,
)
from ..alns.candidates import (
    AttentionCandidatePoolConfig,
    build_attention_repair_candidates,
    build_repair_candidates,
)
from ..alns.node_selection import (
    NodeSelectionName,
    build_best_additional_profile,
    select_next_node,
)


def greedy_repair(
    partial_solution: TOPSolution,
    removed_nodes: list[int],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    rng: random.Random,
) -> TOPSolution:
    """Insert the best feasible node using reward/additional-distance only."""
    del rng  # The baseline uses deterministic tie-breaking for reproducibility.
    result = partial_solution.copy()
    pending = build_repair_candidates(result, removed_nodes, instance)

    while pending:
        best: tuple[float, float, int, int, int] | None = None
        for node_id in sorted(pending):
            reward = instance.nodes[node_id].reward
            for route_index, route in enumerate(result.routes):
                route_distance = calculate_route_distance(
                    route, instance, distance_matrix
                )
                for position in range(1, len(route.node_ids)):
                    additional = insertion_cost(
                        route, position, node_id, distance_matrix
                    )
                    if route_distance + additional > instance.max_distance + 1e-9:
                        continue
                    density = (
                        float("inf")
                        if additional <= 1e-12
                        else reward / additional
                    )
                    candidate = (
                        density,
                        reward,
                        -node_id,
                        -route_index,
                        -position,
                    )
                    if best is None or candidate > best:
                        best = candidate
        if best is None:
            break
        _, _, negative_node, negative_route, negative_position = best
        node_id = -negative_node
        result.routes[-negative_route].node_ids.insert(
            -negative_position, node_id
        )
        pending.remove(node_id)

    return update_solution_metrics(result, instance, distance_matrix)


def controlled_density_repair(
    partial_solution: TOPSolution,
    removed_nodes: list[int],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    rng: random.Random,
    node_selection_strategy: NodeSelectionName = "dynamic_profit_time",
    candidate_pool_config: AttentionCandidatePoolConfig | None = None,
) -> TOPSolution:
    """Fair-control repair using the shared pool and node-selection policy.

    The Attention solver and this baseline differ only in insertion-position
    scoring. For a selected node, this baseline chooses the feasible position
    with the largest reward/additional-distance ratio.
    """
    result = partial_solution.copy()
    route_distances = [
        calculate_route_distance(route, instance, distance_matrix)
        for route in result.routes
    ]
    pending = build_attention_repair_candidates(
        result,
        removed_nodes,
        instance,
        distance_matrix,
        route_distances,
        rng,
        candidate_pool_config,
    )
    best_additional_profile = build_best_additional_profile(
        pending,
        result,
        route_distances,
        instance,
        distance_matrix,
    )
    while pending:
        node_id = select_next_node(
            node_selection_strategy,
            pending,
            removed_nodes,
            result,
            route_distances,
            instance,
            distance_matrix,
            rng,
            best_additional_profile,
        )
        pending.remove(node_id)
        best: tuple[float, float, int, int] | None = None
        for route_index, route in enumerate(result.routes):
            for position in range(1, len(route.node_ids)):
                additional = insertion_cost(
                    route, position, node_id, distance_matrix
                )
                if (
                    route_distances[route_index] + additional
                    > instance.max_distance + 1e-9
                ):
                    continue
                density = (
                    float("inf")
                    if additional <= 1e-12
                    else instance.nodes[node_id].reward / additional
                )
                candidate = (
                    density,
                    -additional,
                    -route_index,
                    -position,
                )
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            continue
        _, negative_additional, negative_route, negative_position = best
        route_index = -negative_route
        position = -negative_position
        result.routes[route_index].node_ids.insert(position, node_id)
        route_distances[route_index] += -negative_additional
        result.visited_nodes.add(node_id)
    return update_solution_metrics(result, instance, distance_matrix)
