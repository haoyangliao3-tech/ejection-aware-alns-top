"""Greedy construction heuristic."""

from __future__ import annotations

from .distance import DistanceMatrix
from .models import Route, TOPInstance, TOPSolution
from .solution import calculate_route_distance, update_solution_metrics


def insertion_cost(
    route: Route, position: int, node_id: int, distance_matrix: DistanceMatrix
) -> float:
    previous_id = route.node_ids[position - 1]
    next_id = route.node_ids[position]
    return (
        distance_matrix[previous_id][node_id]
        + distance_matrix[node_id][next_id]
        - distance_matrix[previous_id][next_id]
    )


def greedy_initial_solution(
    instance: TOPInstance, distance_matrix: DistanceMatrix
) -> TOPSolution:
    routes = [
        Route([instance.depot_id, instance.route_end_id])
        for _ in range(instance.vehicle_count)
    ]
    route_distances = [
        distance_matrix[instance.depot_id][instance.route_end_id]
        for _ in routes
    ]
    unvisited = set(instance.nodes) - instance.depot_ids

    while unvisited:
        best: tuple[float, float, int, int, int] | None = None
        for node_id in sorted(unvisited):
            reward = instance.nodes[node_id].reward
            for route_index, route in enumerate(routes):
                for position in range(1, len(route.node_ids)):
                    additional = insertion_cost(
                        route, position, node_id, distance_matrix
                    )
                    if (
                        route_distances[route_index] + additional
                        > instance.max_distance + 1e-9
                    ):
                        continue
                    score = float("inf") if additional <= 1e-12 else reward / additional
                    candidate = (score, reward, -node_id, -route_index, -position)
                    if best is None or candidate > best:
                        best = candidate
        if best is None:
            break
        _, _, negative_node, negative_route, negative_position = best
        node_id = -negative_node
        route_index = -negative_route
        position = -negative_position
        additional = insertion_cost(
            routes[route_index], position, node_id, distance_matrix
        )
        routes[route_index].node_ids.insert(position, node_id)
        route_distances[route_index] += additional
        unvisited.remove(node_id)

    return update_solution_metrics(
        TOPSolution(routes=routes), instance, distance_matrix
    )
