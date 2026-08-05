"""Implementation-level reproduction of Kim, Li and Johnson (2013).

The implementation follows Algorithms 1--5 in *Expert Systems with
Applications* 40(8), 3065--3072: a solution pool, three ruin criteria,
distance-improving local search, shift-and-insert diversification, and random
plus exhaustive reward replacement.  It is isolated from the proposed
Ejection-Aware ALNS and uses only the shared TOP data structures and distance
utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from time import perf_counter
from typing import Iterable, Sequence

from ..alns.local_search import _compiled_first_move
from ..distance import DistanceMatrix, build_distance_matrix
from ..feasibility import check_solution_feasible
from ..models import Route, TOPInstance, TOPSolution
from ..solution import update_solution_metrics

try:
    import numpy as np
    from numba import njit
except ImportError:  # pragma: no cover - project runtime includes NumPy/Numba.
    np = None
    njit = None


EPSILON = 1e-9


if njit is not None:

    @njit(cache=True)
    def _compiled_replacement_move(
        route: object,
        unrouted: object,
        rewards: object,
        distances: object,
        maximum_distance: float,
        stage: int,
    ) -> object:
        route_size = len(route)
        old_distance = 0.0
        for index in range(route_size - 1):
            old_distance += distances[route[index], route[index + 1]]
        result = np.full(7, -1, dtype=np.int64)
        for removed_position in range(1, route_size - 1):
            removed = route[removed_position]
            base = np.empty(route_size - 1, dtype=np.int64)
            write = 0
            for index in range(route_size):
                if index != removed_position:
                    base[write] = route[index]
                    write += 1
            base_distance = (
                old_distance
                + distances[route[removed_position - 1], route[removed_position + 1]]
                - distances[route[removed_position - 1], removed]
                - distances[removed, route[removed_position + 1]]
            )
            if stage == 1:
                for first_index in range(len(unrouted)):
                    first = unrouted[first_index]
                    reward_gain = rewards[first] - rewards[removed]
                    best_extra = 1.0e300
                    best_position = -1
                    for position in range(1, len(base)):
                        before = base[position - 1]
                        after = base[position]
                        extra = (
                            distances[before, first]
                            + distances[first, after]
                            - distances[before, after]
                        )
                        if extra < best_extra:
                            best_extra = extra
                            best_position = position
                    new_distance = base_distance + best_extra
                    if new_distance <= maximum_distance + 1.0e-9 and (
                        reward_gain > 1.0e-9
                        or (abs(reward_gain) <= 1.0e-9 and new_distance < old_distance - 1.0e-9)
                    ):
                        result[0] = 1
                        result[1] = removed_position
                        result[2] = first
                        result[5] = best_position
                        return result
            else:
                for first_index in range(len(unrouted)):
                    first = unrouted[first_index]
                    for second_index in range(first_index + 1, len(unrouted)):
                        second = unrouted[second_index]
                        reward_gain = rewards[first] + rewards[second] - rewards[removed]
                        if reward_gain < -1.0e-9:
                            continue
                        best_extra = 1.0e300
                        best_kind = -1
                        best_first_position = -1
                        best_second_position = -1
                        second_best_cost = 1.0e300
                        second_best_position = -1
                        second_next_cost = 1.0e300
                        second_next_position = -1
                        for second_position in range(1, len(base)):
                            second_before = base[second_position - 1]
                            second_after = base[second_position]
                            second_cost = (
                                distances[second_before, second]
                                + distances[second, second_after]
                                - distances[second_before, second_after]
                            )
                            if second_cost < second_best_cost:
                                second_next_cost = second_best_cost
                                second_next_position = second_best_position
                                second_best_cost = second_cost
                                second_best_position = second_position
                            elif second_cost < second_next_cost:
                                second_next_cost = second_cost
                                second_next_position = second_position
                        for first_position in range(1, len(base)):
                            before = base[first_position - 1]
                            after = base[first_position]
                            consecutive = (
                                distances[before, first]
                                + distances[first, second]
                                + distances[second, after]
                                - distances[before, after]
                            )
                            if consecutive < best_extra:
                                best_extra = consecutive
                                best_kind = 0
                                best_first_position = first_position
                                best_second_position = first_position
                            reverse = (
                                distances[before, second]
                                + distances[second, first]
                                + distances[first, after]
                                - distances[before, after]
                            )
                            if reverse < best_extra:
                                best_extra = reverse
                                best_kind = 1
                                best_first_position = first_position
                                best_second_position = first_position
                            first_extra = (
                                distances[before, first]
                                + distances[first, after]
                                - distances[before, after]
                            )
                            if second_best_position != first_position:
                                separate = first_extra + second_best_cost
                                separate_position = second_best_position
                            else:
                                separate = first_extra + second_next_cost
                                separate_position = second_next_position
                            if separate_position >= 0 and separate < best_extra:
                                best_extra = separate
                                best_kind = 2
                                best_first_position = first_position
                                best_second_position = separate_position
                        new_distance = base_distance + best_extra
                        if new_distance <= maximum_distance + 1.0e-9 and (
                            reward_gain > 1.0e-9
                            or new_distance < old_distance - 1.0e-9
                        ):
                            result[0] = 2
                            result[1] = removed_position
                            result[2] = first
                            result[3] = second
                            result[4] = best_kind
                            result[5] = best_first_position
                            result[6] = best_second_position
                            return result
        return result

    @njit(cache=True)
    def _compiled_interroute_pair(
        first_route: object,
        second_route: object,
        distances: object,
        maximum_distance: float,
        stage: int,
    ) -> object:
        result = np.full(4, -1, dtype=np.int64)
        first_distance = 0.0
        second_distance = 0.0
        for index in range(len(first_route) - 1):
            first_distance += distances[first_route[index], first_route[index + 1]]
        for index in range(len(second_route) - 1):
            second_distance += distances[second_route[index], second_route[index + 1]]
        old_total = first_distance + second_distance
        if stage == 1:
            for first_position in range(1, len(first_route) - 1):
                first_node = first_route[first_position]
                for second_position in range(1, len(second_route) - 1):
                    second_node = second_route[second_position]
                    first_delta = (
                        distances[first_route[first_position - 1], second_node]
                        + distances[second_node, first_route[first_position + 1]]
                        - distances[first_route[first_position - 1], first_node]
                        - distances[first_node, first_route[first_position + 1]]
                    )
                    second_delta = (
                        distances[second_route[second_position - 1], first_node]
                        + distances[first_node, second_route[second_position + 1]]
                        - distances[second_route[second_position - 1], second_node]
                        - distances[second_node, second_route[second_position + 1]]
                    )
                    if (
                        first_distance + first_delta <= maximum_distance + 1.0e-9
                        and second_distance + second_delta <= maximum_distance + 1.0e-9
                        and first_delta + second_delta < -1.0e-9
                    ):
                        result[0] = first_position
                        result[1] = second_position
                        return result
        elif stage == 2:
            for source_position in range(1, len(first_route) - 1):
                node = first_route[source_position]
                removal = (
                    distances[first_route[source_position - 1], first_route[source_position + 1]]
                    - distances[first_route[source_position - 1], node]
                    - distances[node, first_route[source_position + 1]]
                )
                for target_position in range(1, len(second_route)):
                    insertion = (
                        distances[second_route[target_position - 1], node]
                        + distances[node, second_route[target_position]]
                        - distances[second_route[target_position - 1], second_route[target_position]]
                    )
                    if (
                        second_distance + insertion <= maximum_distance + 1.0e-9
                        and removal + insertion < -1.0e-9
                    ):
                        result[0] = source_position
                        result[1] = target_position
                        return result
        else:
            if len(first_route) < 4 or len(second_route) < 3:
                return result
            for pair_start in range(1, len(first_route) - 2):
                pair_first = first_route[pair_start]
                pair_second = first_route[pair_start + 1]
                pair_removal = (
                    distances[first_route[pair_start - 1], first_route[pair_start + 2]]
                    - distances[first_route[pair_start - 1], pair_first]
                    - distances[pair_first, pair_second]
                    - distances[pair_second, first_route[pair_start + 2]]
                )
                base_first = np.empty(len(first_route) - 2, dtype=np.int64)
                write = 0
                for index in range(len(first_route)):
                    if index != pair_start and index != pair_start + 1:
                        base_first[write] = first_route[index]
                        write += 1
                for singleton_position in range(1, len(second_route) - 1):
                    singleton = second_route[singleton_position]
                    singleton_removal = (
                        distances[second_route[singleton_position - 1], second_route[singleton_position + 1]]
                        - distances[second_route[singleton_position - 1], singleton]
                        - distances[singleton, second_route[singleton_position + 1]]
                    )
                    best_singleton_extra = 1.0e300
                    best_singleton_target = -1
                    for insertion_target in range(1, len(base_first)):
                        extra = (
                            distances[base_first[insertion_target - 1], singleton]
                            + distances[singleton, base_first[insertion_target]]
                            - distances[base_first[insertion_target - 1], base_first[insertion_target]]
                        )
                        if extra < best_singleton_extra:
                            best_singleton_extra = extra
                            best_singleton_target = insertion_target
                    base_second = np.empty(len(second_route) - 1, dtype=np.int64)
                    write = 0
                    for index in range(len(second_route)):
                        if index != singleton_position:
                            base_second[write] = second_route[index]
                            write += 1
                    best_pair_extra = 1.0e300
                    best_pair_target = -1
                    for target in range(1, len(base_second)):
                        extra = (
                            distances[base_second[target - 1], pair_first]
                            + distances[pair_first, pair_second]
                            + distances[pair_second, base_second[target]]
                            - distances[base_second[target - 1], base_second[target]]
                        )
                        if extra < best_pair_extra:
                            best_pair_extra = extra
                            best_pair_target = target
                    new_first = first_distance + pair_removal + best_singleton_extra
                    new_second = second_distance + singleton_removal + best_pair_extra
                    if (
                        new_first <= maximum_distance + 1.0e-9
                        and new_second <= maximum_distance + 1.0e-9
                        and new_first + new_second < old_total - 1.0e-9
                    ):
                        result[0] = pair_start
                        result[1] = singleton_position
                        result[2] = best_singleton_target
                        result[3] = best_pair_target
                        return result
        return result

    @njit(cache=True)
    def _compiled_random_replacement_trial(
        route: object,
        candidates: object,
        deleted: object,
        rewards: object,
        distances: object,
        maximum_distance: float,
    ) -> tuple[object, int, float]:
        work = np.empty(len(route) + len(candidates), dtype=np.int64)
        size = 0
        for index in range(len(route)):
            node = route[index]
            remove = False
            for deleted_index in range(len(deleted)):
                if node == deleted[deleted_index]:
                    remove = True
                    break
            if not remove:
                work[size] = node
                size += 1
        route_distance = 0.0
        for index in range(size - 1):
            route_distance += distances[work[index], work[index + 1]]
        inserted_reward = 0.0
        for candidate_index in range(len(candidates)):
            node = candidates[candidate_index]
            best_extra = 1.0e300
            best_position = -1
            for position in range(1, size):
                extra = (
                    distances[work[position - 1], node]
                    + distances[node, work[position]]
                    - distances[work[position - 1], work[position]]
                )
                if extra < best_extra:
                    best_extra = extra
                    best_position = position
            if route_distance + best_extra <= maximum_distance + 1.0e-9:
                for position in range(size, best_position, -1):
                    work[position] = work[position - 1]
                work[best_position] = node
                size += 1
                route_distance += best_extra
                inserted_reward += rewards[node]
        return work, size, inserted_reward

else:
    _compiled_replacement_move = None
    _compiled_interroute_pair = None
    _compiled_random_replacement_trial = None


_replacement_dense_source: DistanceMatrix | None = None
_replacement_dense_matrix: object = None
_replacement_dense_nodes: tuple[int, ...] = ()
_replacement_dense_index: dict[int, int] = {}
_replacement_rewards_instance: TOPInstance | None = None
_replacement_dense_rewards: object = None


def _ensure_dense_matrix(
    matrix: DistanceMatrix,
) -> tuple[object, tuple[int, ...], dict[int, int]]:
    global _replacement_dense_source, _replacement_dense_matrix
    global _replacement_dense_nodes, _replacement_dense_index
    if np is None:
        raise RuntimeError("NumPy is unavailable")
    if _replacement_dense_source is not matrix:
        ordered = tuple(sorted(matrix))
        _replacement_dense_nodes = ordered
        _replacement_dense_index = {
            node: index for index, node in enumerate(ordered)
        }
        _replacement_dense_matrix = np.asarray(
            [[matrix[first][second] for second in ordered] for first in ordered],
            dtype=np.float64,
        )
        _replacement_dense_source = matrix
    return (
        _replacement_dense_matrix,
        _replacement_dense_nodes,
        _replacement_dense_index,
    )


@dataclass(frozen=True, slots=True)
class KimALNSConfig:
    """Parameters reported by Kim et al.; ``max_iterations`` is experimental."""

    pool_size: int = 20
    maximum_random_deletions: int = 3
    random_replacement_iterations: int = 100

    def validate(self) -> None:
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.maximum_random_deletions <= 0:
            raise ValueError("maximum_random_deletions must be positive")
        if self.random_replacement_iterations <= 0:
            raise ValueError("random_replacement_iterations must be positive")


@dataclass(frozen=True, slots=True)
class KimALNSStats:
    runtime_seconds: float
    completed_iterations: int
    requested_iterations: int
    timed_out: bool
    termination_reason: str
    pool_size: int
    accepted_pool_updates: int


@dataclass(frozen=True, slots=True)
class KimALNSResult:
    solution: TOPSolution
    stats: KimALNSStats


def _expired(deadline: float | None) -> bool:
    return deadline is not None and perf_counter() >= deadline


def _route_distance(nodes: Sequence[int], matrix: DistanceMatrix) -> float:
    return sum(matrix[first][second] for first, second in zip(nodes, nodes[1:]))


def _is_better(candidate: TOPSolution, incumbent: TOPSolution) -> bool:
    return (
        candidate.total_reward > incumbent.total_reward + EPSILON
        or (
            abs(candidate.total_reward - incumbent.total_reward) <= EPSILON
            and candidate.total_distance < incumbent.total_distance - EPSILON
        )
    )


def _signature(solution: TOPSolution) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(route.node_ids) for route in solution.routes)


def _refresh(
    solution: TOPSolution, instance: TOPInstance, matrix: DistanceMatrix
) -> TOPSolution:
    return update_solution_metrics(solution, instance, matrix)


def _unrouted(solution: TOPSolution, instance: TOPInstance) -> set[int]:
    return set(instance.nodes) - instance.depot_ids - solution.visited_nodes


def _insertion_delta(
    nodes: Sequence[int], position: int, node: int, matrix: DistanceMatrix
) -> float:
    before = nodes[position - 1]
    after = nodes[position]
    return matrix[before][node] + matrix[node][after] - matrix[before][after]


def _removal_delta(
    nodes: Sequence[int], position: int, matrix: DistanceMatrix
) -> float:
    before = nodes[position - 1]
    node = nodes[position]
    after = nodes[position + 1]
    return matrix[before][after] - matrix[before][node] - matrix[node][after]


def _best_pair_insertion(
    base: Sequence[int],
    first: int,
    second: int,
    matrix: DistanceMatrix,
) -> tuple[float, list[int]]:
    """Return the exact minimum-distance insertion of two nodes into a route.

    Two inserted nodes either occupy distinct original arcs or occur
    consecutively on one original arc.  Enumerating those cases reduces the
    paper's exhaustive 1-2 replacement from quadratic to linear time in the
    route length without changing its neighbourhood.
    """
    arcs = range(1, len(base))
    first_costs = [
        _insertion_delta(base, position, first, matrix) for position in arcs
    ]
    second_costs = [
        _insertion_delta(base, position, second, matrix) for position in arcs
    ]
    candidates: list[tuple[float, int, int, int]] = []
    # kind 0/1: consecutive insertion on one arc in the two possible orders.
    for position in range(1, len(base)):
        before = base[position - 1]
        after = base[position]
        candidates.append(
            (
                matrix[before][first]
                + matrix[first][second]
                + matrix[second][after]
                - matrix[before][after],
                0,
                position,
                position,
            )
        )
        candidates.append(
            (
                matrix[before][second]
                + matrix[second][first]
                + matrix[first][after]
                - matrix[before][after],
                1,
                position,
                position,
            )
        )
    # kind 2: independent insertions on different original arcs.
    ordered_second = sorted(
        (cost, position)
        for position, cost in zip(range(1, len(base)), second_costs)
    )
    for first_position, first_cost in zip(range(1, len(base)), first_costs):
        for second_cost, second_position in ordered_second[:2]:
            if second_position != first_position:
                candidates.append(
                    (
                        first_cost + second_cost,
                        2,
                        first_position,
                        second_position,
                    )
                )
                break
    extra, kind, first_position, second_position = min(candidates)
    result = list(base)
    if kind == 0:
        result[first_position:first_position] = [first, second]
    elif kind == 1:
        result[first_position:first_position] = [second, first]
    else:
        insertions = sorted(
            ((first_position, first), (second_position, second)), reverse=True
        )
        for position, node in insertions:
            result.insert(position, node)
    return extra, result


def _compiled_replacement_step(
    route: Route,
    unrouted: Sequence[int],
    instance: TOPInstance,
    matrix: DistanceMatrix,
    stage: int,
) -> bool:
    if np is None or _compiled_replacement_move is None or not unrouted:
        return False
    dense_matrix, dense_nodes, index = _ensure_dense_matrix(matrix)
    dense_route = np.asarray([index[node] for node in route.node_ids], dtype=np.int64)
    dense_unrouted = np.asarray([index[node] for node in unrouted], dtype=np.int64)
    global _replacement_rewards_instance, _replacement_dense_rewards
    if _replacement_rewards_instance is not instance:
        _replacement_dense_rewards = np.asarray(
            [instance.nodes[node].reward for node in dense_nodes],
            dtype=np.float64,
        )
        _replacement_rewards_instance = instance
    move = _compiled_replacement_move(
        dense_route,
        dense_unrouted,
        _replacement_dense_rewards,
        dense_matrix,
        float(instance.max_distance),
        stage,
    )
    mode = int(move[0])
    if mode < 0:
        return False
    removed_position = int(move[1])
    first = dense_nodes[int(move[2])]
    base = route.node_ids.copy()
    base.pop(removed_position)
    first_position = int(move[5])
    if mode == 1:
        base.insert(first_position, first)
    else:
        second = dense_nodes[int(move[3])]
        kind = int(move[4])
        second_position = int(move[6])
        if kind == 0:
            base[first_position:first_position] = [first, second]
        elif kind == 1:
            base[first_position:first_position] = [second, first]
        else:
            for position, node in sorted(
                ((first_position, first), (second_position, second)), reverse=True
            ):
                base.insert(position, node)
    route.node_ids[:] = base
    return True


def _best_minimum_distance_insertion(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
    *,
    route_indices: Iterable[int] | None = None,
) -> tuple[int, int, int, float] | None:
    allowed = tuple(route_indices) if route_indices is not None else tuple(
        range(len(solution.routes))
    )
    distances = [_route_distance(route.node_ids, matrix) for route in solution.routes]
    best: tuple[float, int, int, int] | None = None
    for node in sorted(_unrouted(solution, instance)):
        if _expired(deadline):
            break
        for route_index in allowed:
            route = solution.routes[route_index]
            for position in range(1, len(route.node_ids)):
                extra = _insertion_delta(route.node_ids, position, node, matrix)
                if distances[route_index] + extra > instance.max_distance + EPSILON:
                    continue
                key = (extra, node, route_index, position)
                if best is None or key < best:
                    best = key
    if best is None:
        return None
    extra, node, route_index, position = best
    return node, route_index, position, extra


def _insert_as_many_as_possible(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
    *,
    route_indices: Iterable[int] | None = None,
) -> TOPSolution:
    result = _refresh(solution.copy(), instance, matrix)
    while not _expired(deadline):
        move = _best_minimum_distance_insertion(
            result, instance, matrix, deadline, route_indices=route_indices
        )
        if move is None:
            break
        node, route_index, position, _ = move
        result.routes[route_index].node_ids.insert(position, node)
        _refresh(result, instance, matrix)
    return result


def _two_opt_first(nodes: list[int], matrix: DistanceMatrix) -> bool:
    compiled = _compiled_first_move(nodes, matrix)
    if compiled is not None:
        start, end = compiled
        if start < 0:
            return False
        nodes[start : end + 1] = reversed(nodes[start : end + 1])
        return True
    for start in range(1, len(nodes) - 2):
        before = nodes[start - 1]
        first = nodes[start]
        for end in range(start + 1, len(nodes) - 1):
            last = nodes[end]
            after = nodes[end + 1]
            delta = (
                matrix[before][last]
                + matrix[first][after]
                - matrix[before][first]
                - matrix[last][after]
            )
            if delta < -EPSILON:
                nodes[start : end + 1] = reversed(nodes[start : end + 1])
                return True
    return False


def _or_opt_first(nodes: list[int], matrix: DistanceMatrix) -> bool:
    customer_count = len(nodes) - 2
    for length in (1, 2, 3):
        if length > customer_count:
            break
        for start in range(1, len(nodes) - length):
            block = nodes[start : start + length]
            before = nodes[start - 1]
            first = block[0]
            last = block[-1]
            after = nodes[start + length]
            removal = (
                matrix[before][after]
                - matrix[before][first]
                - matrix[last][after]
            )
            remaining = nodes[:start] + nodes[start + length :]
            for target in range(1, len(remaining)):
                insert_before = remaining[target - 1]
                insert_after = remaining[target]
                insertion = (
                    matrix[insert_before][first]
                    + matrix[last][insert_after]
                    - matrix[insert_before][insert_after]
                )
                if removal + insertion < -EPSILON:
                    nodes[:] = remaining[:target] + block + remaining[target:]
                    return True
    return False


def _inter_route_first(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
) -> bool:
    routes = solution.routes
    if np is not None and _compiled_interroute_pair is not None:
        dense_matrix, _, index = _ensure_dense_matrix(matrix)
        dense_routes = [
            np.asarray([index[node] for node in route.node_ids], dtype=np.int64)
            for route in routes
        ]
        for first_index in range(len(routes)):
            for second_index in range(first_index + 1, len(routes)):
                move = _compiled_interroute_pair(
                    dense_routes[first_index],
                    dense_routes[second_index],
                    dense_matrix,
                    float(instance.max_distance),
                    1,
                )
                if int(move[0]) >= 0:
                    first_position = int(move[0])
                    second_position = int(move[1])
                    routes[first_index].node_ids[first_position], routes[
                        second_index
                    ].node_ids[second_position] = (
                        routes[second_index].node_ids[second_position],
                        routes[first_index].node_ids[first_position],
                    )
                    return True
        for source_index in range(len(routes)):
            for target_index in range(len(routes)):
                if target_index == source_index:
                    continue
                move = _compiled_interroute_pair(
                    dense_routes[source_index],
                    dense_routes[target_index],
                    dense_matrix,
                    float(instance.max_distance),
                    2,
                )
                if int(move[0]) >= 0:
                    source_position = int(move[0])
                    target_position = int(move[1])
                    node = routes[source_index].node_ids.pop(source_position)
                    routes[target_index].node_ids.insert(target_position, node)
                    return True
        for first_index in range(len(routes)):
            for second_index in range(len(routes)):
                if first_index == second_index:
                    continue
                move = _compiled_interroute_pair(
                    dense_routes[first_index],
                    dense_routes[second_index],
                    dense_matrix,
                    float(instance.max_distance),
                    3,
                )
                if int(move[0]) >= 0:
                    pair_start = int(move[0])
                    singleton_position = int(move[1])
                    singleton_target = int(move[2])
                    pair_target = int(move[3])
                    pair = routes[first_index].node_ids[
                        pair_start : pair_start + 2
                    ]
                    singleton = routes[second_index].node_ids[singleton_position]
                    base_first = (
                        routes[first_index].node_ids[:pair_start]
                        + routes[first_index].node_ids[pair_start + 2 :]
                    )
                    base_second = routes[second_index].node_ids.copy()
                    base_second.pop(singleton_position)
                    base_first.insert(singleton_target, singleton)
                    base_second[pair_target:pair_target] = pair
                    routes[first_index].node_ids[:] = base_first
                    routes[second_index].node_ids[:] = base_second
                    return True
        return False

    distances = [_route_distance(route.node_ids, matrix) for route in routes]

    # 1-1 exchange.
    for first_route in range(len(routes)):
        for second_route in range(first_route + 1, len(routes)):
            first_nodes = routes[first_route].node_ids
            second_nodes = routes[second_route].node_ids
            for first_position in range(1, len(first_nodes) - 1):
                if _expired(deadline):
                    return False
                for second_position in range(1, len(second_nodes) - 1):
                    first_node = first_nodes[first_position]
                    second_node = second_nodes[second_position]
                    first_delta = (
                        matrix[first_nodes[first_position - 1]][second_node]
                        + matrix[second_node][first_nodes[first_position + 1]]
                        - matrix[first_nodes[first_position - 1]][first_node]
                        - matrix[first_node][first_nodes[first_position + 1]]
                    )
                    second_delta = (
                        matrix[second_nodes[second_position - 1]][first_node]
                        + matrix[first_node][second_nodes[second_position + 1]]
                        - matrix[second_nodes[second_position - 1]][second_node]
                        - matrix[second_node][second_nodes[second_position + 1]]
                    )
                    first_distance = distances[first_route] + first_delta
                    second_distance = distances[second_route] + second_delta
                    if (
                        first_distance <= instance.max_distance + EPSILON
                        and second_distance <= instance.max_distance + EPSILON
                        and first_distance + second_distance
                        < distances[first_route] + distances[second_route] - EPSILON
                    ):
                        first_nodes[first_position] = second_node
                        second_nodes[second_position] = first_node
                        return True

    # 1-0 relocation.
    for source_index, source in enumerate(routes):
        for source_position in range(1, len(source.node_ids) - 1):
            if _expired(deadline):
                return False
            node = source.node_ids[source_position]
            removal = _removal_delta(source.node_ids, source_position, matrix)
            shortened_distance = distances[source_index] + removal
            for target_index, target in enumerate(routes):
                if target_index == source_index:
                    continue
                for target_position in range(1, len(target.node_ids)):
                    insertion = _insertion_delta(
                        target.node_ids, target_position, node, matrix
                    )
                    expanded_distance = distances[target_index] + insertion
                    if (
                        expanded_distance <= instance.max_distance + EPSILON
                        and shortened_distance + expanded_distance
                        < distances[source_index] + distances[target_index] - EPSILON
                    ):
                        source.node_ids.pop(source_position)
                        target.node_ids.insert(target_position, node)
                        return True

    # 2-1 exchange.  Two-stop strings are the standard route-search reading
    # of the paper's 2-1 move and keep the neighbourhood computationally finite.
    for first_index, first in enumerate(routes):
        if len(first.node_ids) < 4:
            continue
        for second_index, second in enumerate(routes):
            if second_index == first_index or len(second.node_ids) < 3:
                continue
            for pair_start in range(1, len(first.node_ids) - 2):
                if _expired(deadline):
                    return False
                pair = first.node_ids[pair_start : pair_start + 2]
                pair_before = first.node_ids[pair_start - 1]
                pair_after = first.node_ids[pair_start + 2]
                pair_removal = (
                    matrix[pair_before][pair_after]
                    - matrix[pair_before][pair[0]]
                    - matrix[pair[0]][pair[1]]
                    - matrix[pair[1]][pair_after]
                )
                for singleton_position in range(1, len(second.node_ids) - 1):
                    singleton = second.node_ids[singleton_position]
                    singleton_removal = _removal_delta(
                        second.node_ids, singleton_position, matrix
                    )
                    base_first = (
                        first.node_ids[:pair_start]
                        + first.node_ids[pair_start + 2 :]
                    )
                    base_second = second.node_ids.copy()
                    base_second.pop(singleton_position)
                    singleton_insertion, singleton_target = min(
                        (
                            _insertion_delta(base_first, target, singleton, matrix),
                            target,
                        )
                        for target in range(1, len(base_first))
                    )
                    pair_insertion, pair_target = min(
                        (
                            matrix[base_second[target - 1]][pair[0]]
                            + matrix[pair[0]][pair[1]]
                            + matrix[pair[1]][base_second[target]]
                            - matrix[base_second[target - 1]][base_second[target]],
                            target,
                        )
                        for target in range(1, len(base_second))
                    )
                    first_distance = (
                        distances[first_index]
                        + pair_removal
                        + singleton_insertion
                    )
                    second_distance = (
                        distances[second_index]
                        + singleton_removal
                        + pair_insertion
                    )
                    if (
                        first_distance <= instance.max_distance + EPSILON
                        and second_distance <= instance.max_distance + EPSILON
                        and first_distance + second_distance
                        < distances[first_index] + distances[second_index] - EPSILON
                    ):
                        first.node_ids[:] = base_first
                        first.node_ids.insert(singleton_target, singleton)
                        second.node_ids[:] = base_second
                        second.node_ids[pair_target:pair_target] = pair
                        return True
    return False


def _local_search(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
) -> TOPSolution:
    result = _refresh(solution.copy(), instance, matrix)
    while not _expired(deadline):
        changed = False
        while not _expired(deadline) and _inter_route_first(
            result, instance, matrix, deadline
        ):
            changed = True
            _refresh(result, instance, matrix)
        intra_changed = True
        while intra_changed and not _expired(deadline):
            intra_changed = False
            for route in result.routes:
                if _two_opt_first(route.node_ids, matrix) or _or_opt_first(
                    route.node_ids, matrix
                ):
                    changed = True
                    intra_changed = True
                    break
        _refresh(result, instance, matrix)
        if not changed:
            break
    return result


def _construct(
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
) -> TOPSolution:
    solution = TOPSolution(
        routes=[
            Route([instance.depot_id, instance.route_end_id])
            for _ in range(instance.vehicle_count)
        ]
    )
    _refresh(solution, instance, matrix)
    for route_index in range(len(solution.routes)):
        route = solution.routes[route_index]
        last = instance.depot_id
        route_distance = matrix[instance.depot_id][instance.route_end_id]
        while not _expired(deadline):
            best: tuple[float, int, float] | None = None
            for node in sorted(_unrouted(solution, instance)):
                reward = instance.nodes[node].reward
                if reward <= 0:
                    continue
                extra = (
                    matrix[last][node]
                    + matrix[node][instance.route_end_id]
                    - matrix[last][instance.route_end_id]
                )
                if route_distance + extra > instance.max_distance + EPSILON:
                    continue
                key = (matrix[last][node] / reward, node, extra)
                if best is None or key < best:
                    best = key
            if best is None:
                break
            _, node, extra = best
            route.node_ids.insert(len(route.node_ids) - 1, node)
            route_distance += extra
            last = node
            _refresh(solution, instance, matrix)
        while not _expired(deadline) and _two_opt_first(route.node_ids, matrix):
            if _expired(deadline):
                break
        _refresh(solution, instance, matrix)
        solution = _insert_as_many_as_possible(
            solution,
            instance,
            matrix,
            deadline,
            route_indices=(route_index,),
        )
    return _refresh(solution, instance, matrix)


def _shift_and_insert(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
) -> TOPSolution:
    result = _refresh(solution.copy(), instance, matrix)
    for source_index in range(len(result.routes)):
        if _expired(deadline):
            break
        moved = False
        original_nodes = result.routes[source_index].node_ids[1:-1].copy()
        for node in original_nodes:
            if _expired(deadline):
                break
            source = result.routes[source_index]
            if node not in source.node_ids:
                continue
            source_position = source.node_ids.index(node)
            shortened = source.node_ids.copy()
            shortened.pop(source_position)
            best: tuple[float, int, int] | None = None
            for target_index, target in enumerate(result.routes):
                if target_index == source_index:
                    continue
                target_distance = _route_distance(target.node_ids, matrix)
                for position in range(1, len(target.node_ids)):
                    extra = _insertion_delta(target.node_ids, position, node, matrix)
                    if target_distance + extra <= instance.max_distance + EPSILON:
                        key = (extra, target_index, position)
                        if best is None or key < best:
                            best = key
            if best is not None:
                _, target_index, position = best
                source.node_ids[:] = shortened
                result.routes[target_index].node_ids.insert(position, node)
                moved = True
                _refresh(result, instance, matrix)
        if moved:
            result = _insert_as_many_as_possible(
                result,
                instance,
                matrix,
                deadline,
                route_indices=(source_index,),
            )
    return _refresh(result, instance, matrix)


def _random_replacement(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    rng: random.Random,
    deadline: float | None,
    config: KimALNSConfig,
) -> TOPSolution:
    result = _refresh(solution.copy(), instance, matrix)
    if np is not None and _compiled_random_replacement_trial is not None:
        dense_matrix, dense_nodes, index = _ensure_dense_matrix(matrix)
        global _replacement_rewards_instance, _replacement_dense_rewards
        if _replacement_rewards_instance is not instance:
            _replacement_dense_rewards = np.asarray(
                [instance.nodes[node].reward for node in dense_nodes],
                dtype=np.float64,
            )
            _replacement_rewards_instance = instance
        for route_index, route in enumerate(result.routes):
            for _ in range(config.random_replacement_iterations):
                if _expired(deadline):
                    return result
                routed = route.node_ids[1:-1]
                unrouted = _unrouted(result, instance)
                if not routed or not unrouted:
                    break
                count = rng.randint(
                    1, min(config.maximum_random_deletions, len(routed))
                )
                deleted = rng.sample(routed, count)
                deleted_reward = sum(
                    instance.nodes[node].reward for node in deleted
                )
                if (
                    sum(instance.nodes[node].reward for node in unrouted)
                    <= deleted_reward
                ):
                    continue
                candidates = list(unrouted | set(deleted))
                rng.shuffle(candidates)
                dense_route = np.asarray(
                    [index[node] for node in route.node_ids], dtype=np.int64
                )
                dense_candidates = np.asarray(
                    [index[node] for node in candidates], dtype=np.int64
                )
                dense_deleted = np.asarray(
                    [index[node] for node in deleted], dtype=np.int64
                )
                trial, size, inserted_reward = _compiled_random_replacement_trial(
                    dense_route,
                    dense_candidates,
                    dense_deleted,
                    _replacement_dense_rewards,
                    dense_matrix,
                    float(instance.max_distance),
                )
                if inserted_reward > deleted_reward + EPSILON:
                    route.node_ids[:] = [
                        dense_nodes[int(trial[position])]
                        for position in range(int(size))
                    ]
                    _refresh(result, instance, matrix)
                    route = result.routes[route_index]
                    break
        return result

    for route_index, route in enumerate(result.routes):
        for _ in range(config.random_replacement_iterations):
            if _expired(deadline):
                return result
            routed = route.node_ids[1:-1]
            unrouted = _unrouted(result, instance)
            if not routed or not unrouted:
                break
            count = rng.randint(1, min(config.maximum_random_deletions, len(routed)))
            deleted = set(rng.sample(routed, count))
            deleted_reward = sum(instance.nodes[node].reward for node in deleted)
            if sum(instance.nodes[node].reward for node in unrouted) <= deleted_reward:
                continue
            candidate = result.copy()
            candidate_route = candidate.routes[route_index]
            candidate_route.node_ids = [
                node for node in candidate_route.node_ids if node not in deleted
            ]
            candidates = list(unrouted | deleted)
            rng.shuffle(candidates)
            inserted: list[int] = []
            distance = _route_distance(candidate_route.node_ids, matrix)
            for node in candidates:
                feasible: list[tuple[float, int]] = []
                for position in range(1, len(candidate_route.node_ids)):
                    extra = _insertion_delta(
                        candidate_route.node_ids, position, node, matrix
                    )
                    if distance + extra <= instance.max_distance + EPSILON:
                        feasible.append((extra, position))
                if feasible:
                    extra, position = min(feasible)
                    candidate_route.node_ids.insert(position, node)
                    distance += extra
                    inserted.append(node)
            inserted_reward = sum(instance.nodes[node].reward for node in inserted)
            if inserted_reward > deleted_reward + EPSILON:
                result = _refresh(candidate, instance, matrix)
                route = result.routes[route_index]
                break
    return result


def _brute_replacement(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    deadline: float | None,
) -> TOPSolution:
    result = _refresh(solution.copy(), instance, matrix)
    if _compiled_replacement_move is not None:
        while not _expired(deadline):
            changed = False
            unrouted = sorted(_unrouted(result, instance))
            for route in result.routes:
                if _compiled_replacement_step(
                    route, unrouted, instance, matrix, stage=1
                ):
                    _refresh(result, instance, matrix)
                    changed = True
                    break
            if changed:
                continue
            unrouted = sorted(_unrouted(result, instance))
            for route in result.routes:
                if _compiled_replacement_step(
                    route, unrouted, instance, matrix, stage=2
                ):
                    _refresh(result, instance, matrix)
                    changed = True
                    break
            if not changed:
                break
        return result

    while not _expired(deadline):
        changed = False
        unrouted = sorted(_unrouted(result, instance))
        for route in result.routes:
            old_distance = _route_distance(route.node_ids, matrix)
            for position in range(1, len(route.node_ids) - 1):
                removed = route.node_ids[position]
                base = route.node_ids.copy()
                base.pop(position)
                base_distance = old_distance + _removal_delta(
                    route.node_ids, position, matrix
                )
                for node in unrouted:
                    if _expired(deadline):
                        return result
                    for target in range(1, len(base)):
                        insertion = _insertion_delta(base, target, node, matrix)
                        distance = base_distance + insertion
                        reward_gain = (
                            instance.nodes[node].reward - instance.nodes[removed].reward
                        )
                        if (
                            distance <= instance.max_distance + EPSILON
                            and (
                                reward_gain > EPSILON
                                or (
                                    abs(reward_gain) <= EPSILON
                                    and distance < old_distance - EPSILON
                                )
                            )
                        ):
                            candidate_nodes = base.copy()
                            candidate_nodes.insert(target, node)
                            route.node_ids[:] = candidate_nodes
                            _refresh(result, instance, matrix)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        if changed:
            continue

        # 1-2 replacement.
        unrouted = sorted(_unrouted(result, instance))
        for route in result.routes:
            old_distance = _route_distance(route.node_ids, matrix)
            for position in range(1, len(route.node_ids) - 1):
                removed = route.node_ids[position]
                base = route.node_ids.copy()
                base.pop(position)
                base_distance = old_distance + _removal_delta(
                    route.node_ids, position, matrix
                )
                for first_index, first in enumerate(unrouted):
                    for second in unrouted[first_index + 1 :]:
                        if _expired(deadline):
                            return result
                        reward_gain = (
                            instance.nodes[first].reward
                            + instance.nodes[second].reward
                            - instance.nodes[removed].reward
                        )
                        if reward_gain < -EPSILON:
                            continue
                        extra, candidate_nodes = _best_pair_insertion(
                            base, first, second, matrix
                        )
                        distance = base_distance + extra
                        if (
                            distance <= instance.max_distance + EPSILON
                            and (
                                reward_gain > EPSILON
                                or distance < old_distance - EPSILON
                            )
                        ):
                            route.node_ids[:] = candidate_nodes
                            _refresh(result, instance, matrix)
                            changed = True
                            break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return result


def _improve_after_ruin(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    rng: random.Random,
    deadline: float | None,
    config: KimALNSConfig,
) -> TOPSolution:
    current = _refresh(solution.copy(), instance, matrix)
    while not _expired(deadline):
        before = current.copy()
        candidate = _local_search(current, instance, matrix, deadline)
        candidate = _shift_and_insert(candidate, instance, matrix, deadline)
        if not _is_better(candidate, before):
            break
        current = candidate
    current = _insert_as_many_as_possible(current, instance, matrix, deadline)
    while not _expired(deadline):
        before = current.copy()
        candidate = _local_search(current, instance, matrix, deadline)
        candidate = _random_replacement(
            candidate, instance, matrix, rng, deadline, config
        )
        candidate = _shift_and_insert(candidate, instance, matrix, deadline)
        candidate = _brute_replacement(candidate, instance, matrix, deadline)
        if not _is_better(candidate, before):
            break
        current = candidate
    return _refresh(current, instance, matrix)


def solve_kim_alns(
    instance: TOPInstance,
    *,
    max_iterations: int = 2500,
    seed: int = 0,
    time_limit_seconds: float | None = None,
    config: KimALNSConfig | None = None,
) -> KimALNSResult:
    """Run the Kim--Li--Johnson augmented LNS reproduction."""
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if time_limit_seconds is not None and time_limit_seconds < 0:
        raise ValueError("time_limit_seconds must be non-negative")
    selected = config or KimALNSConfig()
    selected.validate()
    started = perf_counter()
    deadline = (
        started + time_limit_seconds if time_limit_seconds is not None else None
    )
    rng = random.Random(seed)
    matrix = build_distance_matrix(instance)
    initial = _construct(instance, matrix, deadline)
    initial = _local_search(initial, instance, matrix, deadline)
    best = initial.copy()
    pool = [initial.copy()]
    signatures = {_signature(initial)}
    completed = 0
    accepted_pool_updates = 0

    while completed < max_iterations and not _expired(deadline):
        candidate = rng.choice(pool).copy()
        visited = sorted(candidate.visited_nodes)
        if not visited:
            break
        maximum = max(1, math.floor(0.75 * len(visited)))
        count = rng.randint(1, maximum)
        criterion = rng.randrange(3)
        if criterion == 0:
            removed = set(rng.sample(visited, count))
        elif criterion == 1:
            removed = set(
                sorted(
                    visited,
                    key=lambda node: (-instance.nodes[node].reward, node),
                )[:count]
            )
        else:
            removed = set(
                sorted(
                    visited,
                    key=lambda node: (instance.nodes[node].reward, node),
                )[:count]
            )
        for route in candidate.routes:
            route.node_ids = [node for node in route.node_ids if node not in removed]
        _refresh(candidate, instance, matrix)
        candidate = _improve_after_ruin(
            candidate, instance, matrix, rng, deadline, selected
        )
        if _expired(deadline):
            break
        completed += 1
        if _is_better(candidate, best):
            best = candidate.copy()
        signature = _signature(candidate)
        if signature not in signatures:
            if len(pool) < selected.pool_size:
                pool.append(candidate.copy())
                signatures.add(signature)
                accepted_pool_updates += 1
            else:
                worst_index = min(
                    range(len(pool)),
                    key=lambda index: (
                        pool[index].total_reward,
                        -pool[index].total_distance,
                    ),
                )
                if _is_better(candidate, pool[worst_index]):
                    signatures.remove(_signature(pool[worst_index]))
                    pool[worst_index] = candidate.copy()
                    signatures.add(signature)
                    accepted_pool_updates += 1

    timed_out = _expired(deadline) and completed < max_iterations
    if timed_out:
        reason = "time_limit"
    elif completed >= max_iterations:
        reason = "fixed_iterations"
    else:
        reason = "empty_solution"
    if not check_solution_feasible(best, instance, matrix):
        raise RuntimeError("Kim ALNS reproduction produced an infeasible solution")
    stats = KimALNSStats(
        runtime_seconds=perf_counter() - started,
        completed_iterations=completed,
        requested_iterations=max_iterations,
        timed_out=timed_out,
        termination_reason=reason,
        pool_size=len(pool),
        accepted_pool_updates=accepted_pool_updates,
    )
    return KimALNSResult(solution=best, stats=stats)
