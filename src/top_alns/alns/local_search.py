"""Small, stable local-search interfaces."""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Any

from .attention import AttentionWeights
from .candidates import AttentionCandidatePoolConfig
from .node_selection import NodeSelectionName
from .repair import (
    EjectionRemovalRanking,
    EjectionTelemetry,
    attention_guided_repair,
)
from ..distance import DistanceMatrix
from ..models import Route, TOPInstance, TOPSolution
from ..solution import calculate_route_distance, update_solution_metrics

try:  # Optional compiled acceleration; the pure-Python path remains complete.
    import numpy as np
    from numba import njit
except ImportError:  # pragma: no cover - exercised in minimal installations.
    np = None
    njit = None


if njit is not None:

    @njit(cache=True)
    def _compiled_first_improvement(
        nodes: Any, dense_distances: Any
    ) -> tuple[int, int]:
        """Return the first improving 2-opt interval in reference order."""
        size = len(nodes)
        for start in range(1, size - 2):
            before = nodes[start - 1]
            first = nodes[start]
            for end in range(start + 1, size - 1):
                last = nodes[end]
                after = nodes[end + 1]
                delta = (
                    dense_distances[before, last]
                    + dense_distances[first, after]
                    - dense_distances[before, first]
                    - dense_distances[last, after]
                )
                if delta < -1e-9:
                    return start, end
        return -1, -1

else:
    _compiled_first_improvement = None


_dense_matrix_source: DistanceMatrix | None = None
_dense_matrix: Any = None
_dense_node_index: dict[int, int] | None = None
_dense_uses_direct_ids = False


def _compiled_first_move(
    node_ids: list[int], distance_matrix: DistanceMatrix
) -> tuple[int, int] | None:
    """Return a compiled first-improvement move, or ``None`` to fall back."""
    global _dense_matrix_source, _dense_matrix, _dense_node_index
    global _dense_uses_direct_ids
    if np is None or _compiled_first_improvement is None:
        return None
    if _dense_matrix_source is not distance_matrix:
        ordered = sorted(distance_matrix)
        _dense_node_index = {
            node_id: index for index, node_id in enumerate(ordered)
        }
        _dense_uses_direct_ids = ordered == list(range(len(ordered)))
        _dense_matrix = np.asarray(
            [
                [distance_matrix[first][second] for second in ordered]
                for first in ordered
            ],
            dtype=np.float64,
        )
        _dense_matrix_source = distance_matrix
    assert _dense_node_index is not None
    if _dense_uses_direct_ids:
        dense_nodes = np.asarray(node_ids, dtype=np.int64)
    else:
        dense_nodes = np.fromiter(
            (_dense_node_index[node] for node in node_ids),
            dtype=np.int64,
            count=len(node_ids),
        )
    start, end = _compiled_first_improvement(
        dense_nodes, _dense_matrix
    )
    return int(start), int(end)


def two_opt_route(
    route: Route,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    max_passes: int | None = None,
    *,
    use_fast_path: bool = True,
) -> Route:
    """First-improvement 2-opt.

    ``max_passes`` bounds the number of improving reversals applied. ``None``
    (the default) runs to a local optimum and preserves the original behaviour;
    a small bound is used inside ejection trials, where 2-opt only needs a fast
    feasibility estimate because the chosen route is fully re-optimised later.
    """
    best = route.copy()

    # Ejection trials overwhelmingly request exactly one first-improvement
    # pass.  Keep a dedicated implementation for that case: it follows the
    # same start/end order and evaluates the identical delta expression, but
    # localizes the node list and matrix rows instead of repeatedly resolving
    # dataclass attributes and nested dictionaries.  ``use_fast_path=False``
    # preserves the previous implementation as an executable reference for
    # equivalence tests.
    if max_passes == 1 and use_fast_path:
        nodes = best.node_ids
        compiled_move = _compiled_first_move(nodes, distance_matrix)
        if compiled_move is not None:
            start, end = compiled_move
            if start >= 0:
                nodes[start : end + 1] = reversed(
                    nodes[start : end + 1]
                )
            return best
        matrix = distance_matrix
        for start in range(1, len(nodes) - 2):
            before = nodes[start - 1]
            first = nodes[start]
            before_row = matrix[before]
            first_row = matrix[first]
            before_first = before_row[first]
            for end in range(start + 1, len(nodes) - 1):
                last = nodes[end]
                after = nodes[end + 1]
                delta = (
                    before_row[last]
                    + first_row[after]
                    - before_first
                    - matrix[last][after]
                )
                if delta < -1e-9:
                    nodes[start : end + 1] = reversed(
                        nodes[start : end + 1]
                    )
                    return best
        return best

    improved = True
    passes = 0
    while improved:
        if max_passes is not None and passes >= max_passes:
            break
        passes += 1
        improved = False
        for start in range(1, len(best.node_ids) - 2):
            for end in range(start + 1, len(best.node_ids) - 1):
                before = best.node_ids[start - 1]
                first = best.node_ids[start]
                last = best.node_ids[end]
                after = best.node_ids[end + 1]
                delta = (
                    distance_matrix[before][last]
                    + distance_matrix[first][after]
                    - distance_matrix[before][first]
                    - distance_matrix[last][after]
                )
                if delta < -1e-9:
                    best.node_ids[start : end + 1] = reversed(
                        best.node_ids[start : end + 1]
                    )
                    improved = True
                    break
            if improved:
                break
    return best


def relocate_within_route(
    route: Route, instance: TOPInstance, distance_matrix: DistanceMatrix
) -> Route:
    # TODO: Add best-improvement intra-route relocation.
    del instance, distance_matrix
    return route.copy()


def improve_solution(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> TOPSolution:
    improved = solution.copy()
    improved.routes = [
        two_opt_route(route, instance, distance_matrix)
        for route in improved.routes
    ]
    return update_solution_metrics(improved, instance, distance_matrix)


def improve_then_attention_residual_repair(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    rng: random.Random,
    weights: AttentionWeights,
    node_selection_strategy: NodeSelectionName,
    residual_bucket_size: int,
    candidate_pool_config: AttentionCandidatePoolConfig | None = None,
    enable_ejection: bool = False,
    ejection_max_positions: int = 3,
    ejection_max_nodes: int = 2,
    ejection_max_attempts: int = 100,
    ejection_two_opt_passes: int | None = 1,
    ejection_removal_ranking: EjectionRemovalRanking = "removal_density",
    ejection_telemetry: EjectionTelemetry | None = None,
    ejection_iteration: int | None = None,
    ejection_stage: str = "residual_repair",
) -> TOPSolution:
    """Run 2-opt, then use newly released distance for extra insertions."""
    distance_before_two_opt = solution.total_distance
    improved = improve_solution(solution, instance, distance_matrix)
    if improved.total_distance >= distance_before_two_opt - 1e-9:
        return improved

    if candidate_pool_config is None:
        residual_config = AttentionCandidatePoolConfig(
            bucket_size=max(residual_bucket_size, 1)
        )
    elif candidate_pool_config.bucket_size is None:
        residual_config = replace(
            candidate_pool_config,
            bucket_size=max(residual_bucket_size, 1),
        )
    else:
        residual_config = candidate_pool_config

    return attention_guided_repair(
        improved,
        [],
        instance,
        distance_matrix,
        rng,
        weights,
        node_selection_strategy=node_selection_strategy,
        candidate_pool_config=residual_config,
        enable_ejection=enable_ejection,
        ejection_max_positions=ejection_max_positions,
        ejection_max_nodes=ejection_max_nodes,
        ejection_max_attempts=ejection_max_attempts,
        ejection_two_opt_passes=ejection_two_opt_passes,
        ejection_removal_ranking=ejection_removal_ranking,
        ejection_telemetry=ejection_telemetry,
        ejection_iteration=ejection_iteration,
        ejection_stage=ejection_stage,
    )
