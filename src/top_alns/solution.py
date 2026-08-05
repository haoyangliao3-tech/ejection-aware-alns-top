"""Solution metric helpers."""

from __future__ import annotations

from .distance import DistanceMatrix
from .models import Route, TOPInstance, TOPSolution


def calculate_route_distance(
    route: Route, instance: TOPInstance, distance_matrix: DistanceMatrix
) -> float:
    del instance  # Kept in the stable public interface for future constraints.
    return sum(
        distance_matrix[start][end]
        for start, end in zip(route.node_ids, route.node_ids[1:])
    )


def calculate_solution_distance(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float:
    return sum(
        calculate_route_distance(route, instance, distance_matrix)
        for route in solution.routes
    )


def get_visited_nodes(solution: TOPSolution) -> set[int]:
    return {node_id for route in solution.routes for node_id in route.node_ids}


def calculate_solution_reward(
    solution: TOPSolution, instance: TOPInstance
) -> float:
    visited = get_visited_nodes(solution)
    visited.difference_update(instance.depot_ids)
    return sum(instance.nodes[node_id].reward for node_id in visited)


def update_solution_metrics(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> TOPSolution:
    """Refresh cached aggregate fields and return the same solution."""
    solution.visited_nodes = get_visited_nodes(solution)
    solution.visited_nodes.difference_update(instance.depot_ids)
    solution.total_reward = calculate_solution_reward(solution, instance)
    solution.total_distance = calculate_solution_distance(
        solution, instance, distance_matrix
    )
    return solution
