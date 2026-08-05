"""Feasibility checking with human-readable violations."""

from __future__ import annotations

from typing import TypedDict

from .distance import DistanceMatrix
from .models import Route, TOPInstance, TOPSolution
from .solution import calculate_route_distance

EPSILON = 1e-9


class ValidationResult(TypedDict):
    feasible: bool
    violations: list[str]


def check_route_feasible(
    route: Route, instance: TOPInstance, distance_matrix: DistanceMatrix
) -> bool:
    if len(route.node_ids) < 2:
        return False
    if (
        route.node_ids[0] != instance.depot_id
        or route.node_ids[-1] != instance.route_end_id
    ):
        return False
    if any(node_id not in instance.nodes for node_id in route.node_ids):
        return False
    if any(node in instance.depot_ids for node in route.node_ids[1:-1]):
        return False
    internal = list(route.node_ids[1:-1])
    if len(internal) != len(set(internal)):
        return False
    return (
        calculate_route_distance(route, instance, distance_matrix)
        <= instance.max_distance + EPSILON
    )


def validate_solution(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> ValidationResult:
    violations: list[str] = []
    if len(solution.routes) > instance.vehicle_count:
        violations.append(
            f"route count {len(solution.routes)} exceeds vehicle_count "
            f"{instance.vehicle_count}"
        )

    all_visited: list[int] = []
    for index, route in enumerate(solution.routes):
        if len(route.node_ids) < 2:
            violations.append(f"route {index} must contain start and end depot")
            continue
        if (
            route.node_ids[0] != instance.depot_id
            or route.node_ids[-1] != instance.route_end_id
        ):
            violations.append(f"route {index} must start and end at depot")
        if any(node in instance.depot_ids for node in route.node_ids[1:-1]):
            violations.append(
                f"route {index} contains a depot at an internal position"
            )
        unknown = [node for node in route.node_ids if node not in instance.nodes]
        if unknown:
            violations.append(f"route {index} contains unknown nodes {unknown}")
            continue
        distance = calculate_route_distance(route, instance, distance_matrix)
        if distance > instance.max_distance + EPSILON:
            violations.append(
                f"route {index} distance {distance:.6f} exceeds "
                f"max_distance {instance.max_distance:.6f}"
            )
        all_visited.extend(route.node_ids[1:-1])

    duplicates = sorted(
        {node_id for node_id in all_visited if all_visited.count(node_id) > 1}
    )
    if duplicates:
        violations.append(f"non-depot nodes visited more than once: {duplicates}")
    return {"feasible": not violations, "violations": violations}


def check_solution_feasible(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> bool:
    return validate_solution(solution, instance, distance_matrix)["feasible"]
