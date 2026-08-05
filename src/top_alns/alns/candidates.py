"""Candidate-pool construction shared by repair operators.

The pool definition is shared for a fair comparison. Each repair operator
still applies its own insertion scoring policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from ..distance import DistanceMatrix
from ..models import TOPInstance, TOPSolution
from ..solution import get_visited_nodes


def build_repair_candidates(
    partial_solution: TOPSolution,
    removed_nodes: list[int],
    instance: TOPInstance,
    unvisited_candidate_limit: int | None = None,
) -> set[int]:
    """Return removed nodes plus a selected subset of original unvisited nodes.

    Selection is deterministic and score-neutral with respect to the two
    repair policies: the highest-reward originally unvisited nodes are used.
    By default, the added subset has the same size as ``removed_nodes``.
    """
    currently_visited = get_visited_nodes(partial_solution)
    currently_visited.difference_update(instance.depot_ids)
    removed = (
        set(removed_nodes)
        - currently_visited
        - instance.depot_ids
    )
    originally_unvisited = (
        set(instance.nodes)
        - currently_visited
        - removed
        - instance.depot_ids
    )

    if unvisited_candidate_limit is None:
        limit = max(len(removed), 1)
    else:
        if unvisited_candidate_limit < 0:
            raise ValueError("unvisited_candidate_limit must be non-negative")
        limit = unvisited_candidate_limit

    selected_unvisited = sorted(
        originally_unvisited,
        key=lambda node_id: (
            -instance.nodes[node_id].reward,
            node_id,
        ),
    )[:limit]
    return removed | set(selected_unvisited)


@dataclass(frozen=True, slots=True)
class AttentionCandidatePoolConfig:
    """Sizes of the complementary unvisited-node candidate buckets."""

    bucket_size: int | None = None
    include_reward: bool = True
    include_reward_density: bool = True
    include_spatial: bool = True
    include_regret: bool = True
    include_random: bool = True


def build_attention_repair_candidates(
    partial_solution: TOPSolution,
    removed_nodes: list[int],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    route_distances: list[float],
    rng: random.Random,
    config: AttentionCandidatePoolConfig | None = None,
) -> set[int]:
    """Build a diverse Attention pool instead of a reward-only shortlist."""
    settings = config or AttentionCandidatePoolConfig()
    if settings.bucket_size is not None and settings.bucket_size < 0:
        raise ValueError("bucket_size must be non-negative")

    currently_visited = get_visited_nodes(partial_solution)
    currently_visited.difference_update(instance.depot_ids)
    removed = (
        set(removed_nodes)
        - currently_visited
        - instance.depot_ids
    )
    unvisited = sorted(
        set(instance.nodes)
        - currently_visited
        - removed
        - instance.depot_ids
    )
    bucket_size = (
        max(len(removed), 1)
        if settings.bucket_size is None
        else settings.bucket_size
    )
    if bucket_size == 0 or not unvisited:
        return removed

    selected: set[int] = set()
    if settings.include_reward:
        selected.update(
            sorted(
                unvisited,
                key=lambda node: (
                    -instance.nodes[node].reward,
                    node,
                ),
            )[:bucket_size]
        )

    profiles: dict[int, tuple[float, float, float]] = {}
    for node in unvisited:
        best_additional = float("inf")
        second_additional = float("inf")
        minimum_neighbor_distance = float("inf")
        for route_index, route in enumerate(partial_solution.routes):
            for position in range(1, len(route.node_ids)):
                previous = route.node_ids[position - 1]
                following = route.node_ids[position]
                previous_distance = distance_matrix[previous][node]
                next_distance = distance_matrix[node][following]
                additional = (
                    previous_distance
                    + next_distance
                    - distance_matrix[previous][following]
                )
                if (
                    route_distances[route_index] + additional
                    <= instance.max_distance + 1e-9
                ):
                    if additional < best_additional:
                        second_additional = best_additional
                        best_additional = additional
                    elif additional < second_additional:
                        second_additional = additional
                    minimum_neighbor_distance = min(
                        minimum_neighbor_distance,
                        (previous_distance + next_distance) / 2.0,
                    )
        if best_additional < float("inf"):
            effective_second = (
                second_additional
                if second_additional < float("inf")
                else instance.max_distance
            )
            profiles[node] = (
                instance.nodes[node].reward
                / max(best_additional, 1e-12),
                minimum_neighbor_distance,
                max(0.0, effective_second - best_additional),
            )

    feasible_nodes = list(profiles)
    if settings.include_reward_density:
        selected.update(
            sorted(
                feasible_nodes,
                key=lambda node: (-profiles[node][0], node),
            )[:bucket_size]
        )
    if settings.include_spatial:
        selected.update(
            sorted(
                feasible_nodes,
                key=lambda node: (profiles[node][1], node),
            )[:bucket_size]
        )
    if settings.include_regret:
        selected.update(
            sorted(
                feasible_nodes,
                key=lambda node: (-profiles[node][2], node),
            )[:bucket_size]
        )
    if settings.include_random:
        selected.update(
            rng.sample(unvisited, min(bucket_size, len(unvisited)))
        )
    return removed | selected
