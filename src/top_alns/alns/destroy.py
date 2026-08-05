"""Basic destroy operators."""

from __future__ import annotations

import math
import random

from ..distance import DistanceMatrix
from ..models import TOPInstance, TOPSolution
from ..solution import update_solution_metrics


def _remove_nodes(solution: TOPSolution, node_ids: set[int]) -> TOPSolution:
    result = solution.copy()
    for route in result.routes:
        route.node_ids = [node for node in route.node_ids if node not in node_ids]
    result.visited_nodes = solution.visited_nodes - node_ids
    return result


def random_removal(
    solution: TOPSolution, remove_count: int, rng: random.Random
) -> tuple[TOPSolution, list[int]]:
    candidates = sorted(solution.visited_nodes)
    count = min(max(remove_count, 0), len(candidates))
    removed = rng.sample(candidates, count)
    return _remove_nodes(solution, set(removed)), removed


def low_reward_density_removal(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    remove_count: int,
    rng: random.Random,
) -> tuple[TOPSolution, list[int]]:
    del rng  # Stable signature; deterministic ties are useful in this baseline.
    densities: list[tuple[float, int]] = []
    for route in solution.routes:
        for position in range(1, len(route.node_ids) - 1):
            node_id = route.node_ids[position]
            previous_id = route.node_ids[position - 1]
            next_id = route.node_ids[position + 1]
            marginal = (
                distance_matrix[previous_id][node_id]
                + distance_matrix[node_id][next_id]
                - distance_matrix[previous_id][next_id]
            )
            density = instance.nodes[node_id].reward / max(marginal, 1e-12)
            densities.append((density, node_id))
    densities.sort(key=lambda item: (item[0], item[1]))
    removed = [node_id for _, node_id in densities[: max(remove_count, 0)]]
    partial = _remove_nodes(solution, set(removed))
    update_solution_metrics(partial, instance, distance_matrix)
    return partial, removed


def largest_saving_removal(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    remove_count: int,
    rng: random.Random,
) -> tuple[TOPSolution, list[int]]:
    """Prefer nodes whose removal releases the most route distance."""
    savings: list[tuple[int, float]] = []
    for route in solution.routes:
        for position in range(1, len(route.node_ids) - 1):
            node_id = route.node_ids[position]
            previous_id = route.node_ids[position - 1]
            next_id = route.node_ids[position + 1]
            saving = (
                distance_matrix[previous_id][node_id]
                + distance_matrix[node_id][next_id]
                - distance_matrix[previous_id][next_id]
            )
            savings.append((node_id, max(saving, 1e-12)))

    count = min(max(remove_count, 0), len(savings))
    remaining = savings.copy()
    removed: list[int] = []
    for _ in range(count):
        index = rng.choices(
            range(len(remaining)),
            weights=[saving for _, saving in remaining],
            k=1,
        )[0]
        node_id, _ = remaining.pop(index)
        removed.append(node_id)
    partial = _remove_nodes(solution, set(removed))
    update_solution_metrics(partial, instance, distance_matrix)
    return partial, removed


def sequence_removal(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    remove_count: int,
    rng: random.Random,
) -> tuple[TOPSolution, list[int]]:
    """Remove one connected sequence to create a large insertion slot."""
    eligible = [
        route
        for route in solution.routes
        if len(route.node_ids) - 2 >= 2
    ]
    if not eligible or remove_count <= 0:
        return solution.copy(), []
    route = rng.choice(eligible)
    customer_count = len(route.node_ids) - 2
    maximum_length = max(
        2,
        min(
            customer_count,
            max(remove_count, 2),
            math.ceil(0.15 * customer_count),
        ),
    )
    length = rng.randint(2, maximum_length)
    start = rng.randint(1, customer_count - length + 1)
    removed = route.node_ids[start : start + length].copy()
    partial = _remove_nodes(solution, set(removed))
    update_solution_metrics(partial, instance, distance_matrix)
    return partial, removed


def route_removal(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    remove_count: int,
    rng: random.Random,
) -> tuple[TOPSolution, list[int]]:
    """Remove every customer from one randomly selected non-empty route."""
    del remove_count
    eligible = [
        route
        for route in solution.routes
        if len(route.node_ids) > 2
    ]
    if not eligible:
        return solution.copy(), []
    removed = rng.choice(eligible).node_ids[1:-1].copy()
    partial = _remove_nodes(solution, set(removed))
    update_solution_metrics(partial, instance, distance_matrix)
    return partial, removed
