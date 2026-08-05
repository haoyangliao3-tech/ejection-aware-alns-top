"""Node-selection strategies kept separate from insertion-position scoring."""

from __future__ import annotations

import random
from typing import Literal

from ..distance import DistanceMatrix
from ..models import TOPInstance, TOPSolution

NodeSelectionName = Literal[
    "dynamic_profit_time",
    "highest_profit",
    "random",
    "lrfi",
]


def _best_feasible_additional(
    node_id: int,
    solution: TOPSolution,
    route_distances: list[float],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float | None:
    best: float | None = None
    for route_index, route in enumerate(solution.routes):
        for position in range(1, len(route.node_ids)):
            previous = route.node_ids[position - 1]
            following = route.node_ids[position]
            additional = (
                distance_matrix[previous][node_id]
                + distance_matrix[node_id][following]
                - distance_matrix[previous][following]
            )
            if (
                route_distances[route_index] + additional
                <= instance.max_distance + 1e-9
                and (best is None or additional < best)
            ):
                best = additional
    return best


def build_best_additional_profile(
    pending: set[int],
    solution: TOPSolution,
    route_distances: list[float],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> dict[int, float | None]:
    """Snapshot each node's best feasible insertion cost for node ordering."""
    return {
        node: _best_feasible_additional(
            node,
            solution,
            route_distances,
            instance,
            distance_matrix,
        )
        for node in pending
    }


def select_next_node(
    strategy: NodeSelectionName,
    pending: set[int],
    removed_order: list[int],
    solution: TOPSolution,
    route_distances: list[float],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    rng: random.Random,
    best_additional_profile: dict[int, float | None] | None = None,
) -> int:
    """Select a node without deciding its route or insertion position."""
    if not pending:
        raise ValueError("pending must not be empty")
    ordered = sorted(pending)
    if strategy == "random":
        return rng.choice(ordered)
    if strategy == "highest_profit":
        weights = [
            max(instance.nodes[node].reward, 1e-12)
            for node in ordered
        ]
        return rng.choices(ordered, weights=weights, k=1)[0]
    if strategy == "lrfi":
        for node in reversed(removed_order):
            if node in pending:
                return node
        return rng.choice(ordered)
    if strategy == "dynamic_profit_time":
        scored: list[tuple[float, float, int]] = []
        for node in ordered:
            additional = (
                best_additional_profile.get(node)
                if best_additional_profile is not None
                else _best_feasible_additional(
                    node,
                    solution,
                    route_distances,
                    instance,
                    distance_matrix,
                )
            )
            score = (
                -1.0
                if additional is None
                else instance.nodes[node].reward
                / max(additional, 1e-12)
            )
            scored.append(
                (score, instance.nodes[node].reward, -node)
            )
        return -max(scored)[2]
    raise ValueError(f"unknown node selection strategy: {strategy}")
