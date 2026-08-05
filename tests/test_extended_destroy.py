import random

from top_alns.alns.destroy import (
    largest_saving_removal,
    route_removal,
    sequence_removal,
)
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.models import Node, Route, TOPInstance, TOPSolution
from top_alns.solution import update_solution_metrics


def _problem() -> tuple[TOPInstance, TOPSolution, dict[int, dict[int, float]]]:
    instance = TOPInstance(
        nodes={
            0: Node(0, 0.0, 0.0, 0.0),
            1: Node(1, 1.0, 1.0, 10.0),
            2: Node(2, 2.0, -1.0, 20.0),
            3: Node(3, 3.0, 1.0, 30.0),
            4: Node(4, -1.0, 1.0, 40.0),
            5: Node(5, -2.0, -1.0, 50.0),
        },
        depot_id=0,
        vehicle_count=2,
        max_distance=30.0,
    )
    matrix = build_distance_matrix(instance)
    solution = update_solution_metrics(
        TOPSolution(
            [
                Route([0, 1, 2, 3, 0]),
                Route([0, 4, 5, 0]),
            ]
        ),
        instance,
        matrix,
    )
    return instance, solution, matrix


def test_largest_saving_removal_removes_requested_count() -> None:
    instance, solution, matrix = _problem()
    partial, removed = largest_saving_removal(
        solution, instance, matrix, 2, random.Random(0)
    )
    assert len(removed) == 2
    assert set(removed).isdisjoint(partial.visited_nodes)
    assert check_solution_feasible(partial, instance, matrix)


def test_sequence_removal_is_connected_in_original_route() -> None:
    instance, solution, matrix = _problem()
    partial, removed = sequence_removal(
        solution, instance, matrix, 2, random.Random(0)
    )
    assert len(removed) == 2
    assert any(
        removed == route.node_ids[start : start + len(removed)]
        for route in solution.routes
        for start in range(1, len(route.node_ids) - len(removed))
    )
    assert check_solution_feasible(partial, instance, matrix)


def test_route_removal_clears_exactly_one_route() -> None:
    instance, solution, matrix = _problem()
    partial, removed = route_removal(
        solution, instance, matrix, 2, random.Random(0)
    )
    original_customer_sets = [
        set(route.node_ids[1:-1]) for route in solution.routes
    ]
    assert set(removed) in original_customer_sets
    assert set(removed).isdisjoint(partial.visited_nodes)
    assert check_solution_feasible(partial, instance, matrix)
