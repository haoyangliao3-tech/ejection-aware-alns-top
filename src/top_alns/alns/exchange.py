"""Prize-aware compound exchanges used to cross tight feasibility walls."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..distance import DistanceMatrix
from ..models import Route, TOPInstance, TOPSolution
from ..solution import calculate_route_distance, update_solution_metrics
from .local_search import two_opt_route


@dataclass(frozen=True, slots=True)
class ExchangeResult:
    solution: TOPSolution
    ejected_nodes: list[int]
    attempted: bool


def _removal_density(
    route: Route,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> tuple[float, float, int]:
    node = route.node_ids[position]
    previous = route.node_ids[position - 1]
    following = route.node_ids[position + 1]
    saving = (
        distance_matrix[previous][node]
        + distance_matrix[node][following]
        - distance_matrix[previous][following]
    )
    return (
        instance.nodes[node].reward / max(saving, 1e-12),
        instance.nodes[node].reward,
        node,
    )


def prize_collecting_exchange(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    *,
    top_unvisited: int = 20,
    positions_per_node: int = 3,
    ejection_pool_size: int = 15,
    max_ejections: int = 2,
) -> ExchangeResult:
    """Find an improving 1-in/0..2-out move by bounded enumeration.

    The inserted node may initially violate the route budget. Low marginal
    value nodes from that route are then ejected jointly, and the resulting
    route is improved with 2-opt before exact feasibility is checked.
    """
    if (
        top_unvisited <= 0
        or positions_per_node <= 0
        or ejection_pool_size <= 0
        or max_ejections < 0
    ):
        return ExchangeResult(solution.copy(), [], False)

    unvisited = sorted(
        set(instance.nodes) - solution.visited_nodes - instance.depot_ids,
        key=lambda node: (-instance.nodes[node].reward, node),
    )[:top_unvisited]
    if not unvisited:
        return ExchangeResult(solution.copy(), [], False)

    route_distances = [
        calculate_route_distance(route, instance, distance_matrix)
        for route in solution.routes
    ]
    best_reward = solution.total_reward
    best_distance = solution.total_distance
    best_route: Route | None = None
    best_route_index = -1
    best_ejected: list[int] = []

    for new_node in unvisited:
        forced_positions: list[tuple[float, int, int]] = []
        for route_index, route in enumerate(solution.routes):
            for position in range(1, len(route.node_ids)):
                previous = route.node_ids[position - 1]
                following = route.node_ids[position]
                additional = (
                    distance_matrix[previous][new_node]
                    + distance_matrix[new_node][following]
                    - distance_matrix[previous][following]
                )
                forced_positions.append((additional, route_index, position))

        for _, route_index, position in sorted(forced_positions)[
            :positions_per_node
        ]:
            original_route = solution.routes[route_index]
            removable = sorted(
                range(1, len(original_route.node_ids) - 1),
                key=lambda index: _removal_density(
                    original_route,
                    index,
                    instance,
                    distance_matrix,
                ),
            )[:ejection_pool_size]
            removable_nodes = [
                original_route.node_ids[index] for index in removable
            ]
            maximum = min(max_ejections, len(removable_nodes))
            for count in range(maximum + 1):
                for ejected_tuple in combinations(removable_nodes, count):
                    ejected = set(ejected_tuple)
                    reward = (
                        solution.total_reward
                        + instance.nodes[new_node].reward
                        - sum(instance.nodes[node].reward for node in ejected)
                    )
                    if reward <= best_reward + 1e-9:
                        continue

                    route_nodes = original_route.node_ids.copy()
                    route_nodes.insert(position, new_node)
                    route_nodes = [
                        node for node in route_nodes if node not in ejected
                    ]
                    candidate_route = two_opt_route(
                        Route(route_nodes), instance, distance_matrix
                    )
                    new_route_distance = calculate_route_distance(
                        candidate_route, instance, distance_matrix
                    )
                    if new_route_distance > instance.max_distance + 1e-9:
                        continue
                    total_distance = (
                        solution.total_distance
                        - route_distances[route_index]
                        + new_route_distance
                    )
                    if (
                        reward > best_reward + 1e-9
                        or (
                            abs(reward - best_reward) <= 1e-9
                            and total_distance < best_distance - 1e-9
                        )
                    ):
                        best_reward = reward
                        best_distance = total_distance
                        best_route = candidate_route
                        best_route_index = route_index
                        best_ejected = list(ejected_tuple)

    if best_route is None:
        return ExchangeResult(solution.copy(), [], True)
    result = solution.copy()
    result.routes[best_route_index] = best_route
    update_solution_metrics(result, instance, distance_matrix)
    return ExchangeResult(result, best_ejected, True)
