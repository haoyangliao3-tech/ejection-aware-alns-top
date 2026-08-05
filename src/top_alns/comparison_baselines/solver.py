"""GRASP, iterated local search and variable-neighbourhood baselines.

These implementations intentionally use only classical reward/distance greedy
insertion, elementary ruin operators and bounded 2-opt.  They do not call the
attention score, ejection repair, adaptive operator weights or route pool used
by the proposed method.

For GRASP, one budget unit is one randomized constructive insertion (or one
restart when no insertion is feasible).  Defining the budget at this elementary
step level keeps a 2,500-unit run practical on 400-node instances; treating one
unit as a complete GRASP reconstruction would not be computationally comparable.
For ILS and VNS, one budget unit is one perturbation-repair iteration.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import random
from time import perf_counter
from typing import Literal

from ..alns.destroy import (
    low_reward_density_removal,
    random_removal,
    sequence_removal,
)
from ..alns.local_search import two_opt_route
from ..distance import DistanceMatrix, build_distance_matrix
from ..feasibility import check_solution_feasible
from ..greedy import greedy_initial_solution, insertion_cost
from ..models import Route, TOPInstance, TOPSolution
from ..solution import update_solution_metrics
from ..vanilla_alns.repair import greedy_repair

ComparisonAlgorithm = Literal["grasp", "ils", "vns"]
ProgressCallback = Callable[[int, int], None]


def _is_better(candidate: TOPSolution, incumbent: TOPSolution) -> bool:
    return (
        candidate.total_reward > incumbent.total_reward + 1e-9
        or (
            abs(candidate.total_reward - incumbent.total_reward) <= 1e-9
            and candidate.total_distance < incumbent.total_distance - 1e-9
        )
    )


def _bounded_two_opt(
    solution: TOPSolution,
    instance: TOPInstance,
    matrix: DistanceMatrix,
    max_passes: int = 2,
) -> TOPSolution:
    result = solution.copy()
    result.routes = [
        two_opt_route(route, instance, matrix, max_passes=max_passes)
        for route in result.routes
    ]
    return update_solution_metrics(result, instance, matrix)


def _empty_solution(
    instance: TOPInstance, matrix: DistanceMatrix
) -> tuple[TOPSolution, list[float], set[int]]:
    routes = [
        Route([instance.depot_id, instance.route_end_id])
        for _ in range(instance.vehicle_count)
    ]
    base_distance = matrix[instance.depot_id][instance.route_end_id]
    return TOPSolution(routes=routes), [base_distance] * len(routes), (
        set(instance.nodes) - instance.depot_ids
    )


def _best_position(
    node_id: int,
    solution: TOPSolution,
    route_distances: list[float],
    instance: TOPInstance,
    matrix: DistanceMatrix,
) -> tuple[float, int, int, float] | None:
    best: tuple[float, int, int, float] | None = None
    reward = instance.nodes[node_id].reward
    for route_index, route in enumerate(solution.routes):
        for position in range(1, len(route.node_ids)):
            additional = insertion_cost(route, position, node_id, matrix)
            if route_distances[route_index] + additional > instance.max_distance + 1e-9:
                continue
            score = reward / max(additional, 1e-12)
            candidate = (score, route_index, position, additional)
            if best is None or (score, -additional, -route_index, -position) > (
                best[0], -best[3], -best[1], -best[2]
            ):
                best = candidate
    return best


def _grasp(
    instance: TOPInstance,
    matrix: DistanceMatrix,
    iterations: int,
    rng: random.Random,
    progress: ProgressCallback | None,
    time_limit_seconds: float | None = None,
    alpha: float = 0.20,
    sample_size: int = 32,
) -> TOPSolution:
    """GRASP with a restricted candidate list and repeated reconstruction."""
    current, route_distances, unvisited = _empty_solution(instance, matrix)
    best = update_solution_metrics(current.copy(), instance, matrix)
    if progress:
        progress(0, iterations)
    deadline = (
        perf_counter() + time_limit_seconds
        if time_limit_seconds is not None
        else None
    )

    stopped_by_time_limit = False
    for iteration in range(iterations):
        if deadline is not None and perf_counter() >= deadline:
            stopped_by_time_limit = True
            break
        sample = (
            sorted(unvisited)
            if len(unvisited) <= sample_size
            else rng.sample(sorted(unvisited), sample_size)
        )
        candidates: list[tuple[int, float, int, int, float]] = []
        for node_id in sample:
            placement = _best_position(
                node_id, current, route_distances, instance, matrix
            )
            if placement is not None:
                score, route_index, position, additional = placement
                candidates.append(
                    (node_id, score, route_index, position, additional)
                )

        # A random sample can miss the remaining feasible nodes.  Before a
        # restart, scan the complete set once to avoid premature termination.
        if not candidates and len(sample) < len(unvisited):
            for node_id in sorted(unvisited):
                placement = _best_position(
                    node_id, current, route_distances, instance, matrix
                )
                if placement is not None:
                    score, route_index, position, additional = placement
                    candidates.append(
                        (node_id, score, route_index, position, additional)
                    )

        if candidates:
            scores = [item[1] for item in candidates]
            lower = min(scores)
            threshold = max(scores) - alpha * (max(scores) - lower)
            restricted = [item for item in candidates if item[1] >= threshold - 1e-12]
            node_id, _, route_index, position, additional = rng.choice(restricted)
            current.routes[route_index].node_ids.insert(position, node_id)
            route_distances[route_index] += additional
            unvisited.remove(node_id)
        else:
            completed = (
                update_solution_metrics(current.copy(), instance, matrix)
                if deadline is not None and perf_counter() >= deadline
                else _bounded_two_opt(current, instance, matrix)
            )
            if _is_better(completed, best):
                best = completed.copy()
            current, route_distances, unvisited = _empty_solution(instance, matrix)

        if progress:
            progress(iteration + 1, iterations)

    completed = (
        update_solution_metrics(current.copy(), instance, matrix)
        if stopped_by_time_limit
        or (deadline is not None and perf_counter() >= deadline)
        else _bounded_two_opt(current, instance, matrix)
    )
    if _is_better(completed, best):
        best = completed
    return best


def _ils(
    instance: TOPInstance,
    matrix: DistanceMatrix,
    iterations: int,
    rng: random.Random,
    progress: ProgressCallback | None,
    removal_rate: float = 0.10,
    time_limit_seconds: float | None = None,
) -> TOPSolution:
    deadline = (
        perf_counter() + time_limit_seconds
        if time_limit_seconds is not None
        else None
    )
    current = _bounded_two_opt(greedy_initial_solution(instance, matrix), instance, matrix)
    best = current.copy()
    if progress:
        progress(0, iterations)
    maximum_iteration_seconds = 0.0
    for iteration in range(iterations):
        iteration_started = perf_counter()
        if deadline is not None:
            remaining = deadline - iteration_started
            if remaining <= max(0.002, 1.25 * maximum_iteration_seconds):
                break
        if not current.visited_nodes:
            break
        remove_count = max(1, math.ceil(len(current.visited_nodes) * removal_rate))
        partial, removed = random_removal(current, remove_count, rng)
        update_solution_metrics(partial, instance, matrix)
        candidate = greedy_repair(partial, removed, instance, matrix, rng)
        candidate = _bounded_two_opt(candidate, instance, matrix)
        if check_solution_feasible(candidate, instance, matrix):
            # Standard ILS walks between local optima; the incumbent best is
            # retained separately and used for periodic intensification.
            current = candidate
            if _is_better(candidate, best):
                best = candidate.copy()
            elif (iteration + 1) % 50 == 0:
                current = best.copy()
        if progress:
            progress(iteration + 1, iterations)
        maximum_iteration_seconds = max(
            maximum_iteration_seconds,
            perf_counter() - iteration_started,
        )
    return best


def _vns(
    instance: TOPInstance,
    matrix: DistanceMatrix,
    iterations: int,
    rng: random.Random,
    progress: ProgressCallback | None,
    time_limit_seconds: float | None = None,
) -> TOPSolution:
    deadline = (
        perf_counter() + time_limit_seconds
        if time_limit_seconds is not None
        else None
    )
    current = _bounded_two_opt(greedy_initial_solution(instance, matrix), instance, matrix)
    best = current.copy()
    neighbourhood = 0
    if progress:
        progress(0, iterations)
    maximum_iteration_seconds = 0.0
    for iteration in range(iterations):
        iteration_started = perf_counter()
        if deadline is not None:
            remaining = deadline - iteration_started
            if remaining <= max(0.002, 1.25 * maximum_iteration_seconds):
                break
        if not current.visited_nodes:
            break
        rates = (0.05, 0.10, 0.20)
        remove_count = max(1, math.ceil(len(current.visited_nodes) * rates[neighbourhood]))
        if neighbourhood == 0:
            partial, removed = random_removal(current, remove_count, rng)
        elif neighbourhood == 1:
            partial, removed = low_reward_density_removal(
                current, instance, matrix, remove_count, rng
            )
        else:
            partial, removed = sequence_removal(
                current, instance, matrix, remove_count, rng
            )
        candidate = greedy_repair(partial, removed, instance, matrix, rng)
        candidate = _bounded_two_opt(candidate, instance, matrix)
        if check_solution_feasible(candidate, instance, matrix) and _is_better(candidate, current):
            current = candidate
            if _is_better(candidate, best):
                best = candidate.copy()
            neighbourhood = 0
        else:
            neighbourhood = (neighbourhood + 1) % 3
        if progress:
            progress(iteration + 1, iterations)
        maximum_iteration_seconds = max(
            maximum_iteration_seconds,
            perf_counter() - iteration_started,
        )
    return best


def solve_comparison_baseline(
    instance: TOPInstance,
    algorithm: ComparisonAlgorithm,
    *,
    max_iterations: int = 2500,
    seed: int = 0,
    progress_callback: ProgressCallback | None = None,
    time_limit_seconds: float | None = None,
) -> TOPSolution:
    """Solve one instance using one of the three independent baselines."""
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if time_limit_seconds is not None and time_limit_seconds < 0.0:
        raise ValueError("time_limit_seconds must be non-negative")
    solve_started = perf_counter()
    matrix = build_distance_matrix(instance)
    rng = random.Random(seed)
    remaining_time = (
        max(0.0, time_limit_seconds - (perf_counter() - solve_started))
        if time_limit_seconds is not None
        else None
    )
    if algorithm == "grasp":
        result = _grasp(
            instance,
            matrix,
            max_iterations,
            rng,
            progress_callback,
            time_limit_seconds=remaining_time,
        )
    elif algorithm == "ils":
        result = _ils(
            instance,
            matrix,
            max_iterations,
            rng,
            progress_callback,
            time_limit_seconds=remaining_time,
        )
    elif algorithm == "vns":
        result = _vns(
            instance,
            matrix,
            max_iterations,
            rng,
            progress_callback,
            time_limit_seconds=remaining_time,
        )
    else:
        raise ValueError("algorithm must be 'grasp', 'ils', or 'vns'")
    if not check_solution_feasible(result, instance, matrix):
        raise RuntimeError(f"{algorithm} produced an infeasible solution")
    return result
