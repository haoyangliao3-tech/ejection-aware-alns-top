"""Basic repair operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from time import perf_counter
from typing import Literal

from ..distance import DistanceMatrix
from ..greedy import insertion_cost
from ..models import Route, TOPInstance, TOPSolution
from ..solution import (
    calculate_route_distance,
    update_solution_metrics,
)
from .attention import (
    AttentionRouteCache,
    AttentionScoreComponents,
    AttentionWeights,
    incremental_attention_score_components,
    normalize_attention_components,
)
from .candidates import (
    AttentionCandidatePoolConfig,
    build_repair_candidates,
    build_attention_repair_candidates,
)
from .node_selection import (
    NodeSelectionName,
    build_best_additional_profile,
    select_next_node,
)


EjectionRemovalRanking = Literal["removal_density", "random"]


@dataclass
class EjectionCaseRecord:
    """Compact, JSON-ready evidence for one real repair attempt."""

    outcome: str
    iteration: int | None
    stage: str
    blocked_customer: int
    route_index: int | None
    original_route: list[int]
    inserted_route: list[int]
    post_ejection_route: list[int]
    final_route: list[int]
    original_distance: float | None
    inserted_distance: float | None
    post_ejection_distance: float | None
    final_distance: float | None
    distance_limit: float
    insertion_excess: float | None
    ejected_customers: list[int]
    ejection_count: int
    bounded_two_opt_attempted: bool
    bounded_two_opt_changed_route: bool
    net_reward_gain: float | None
    failure_reason: str | None = None
    produced_new_best: bool = False
    commit_ordinal: int | None = None


@dataclass
class EjectionTelemetry:
    """Passive counters for the bounded repair-level ejection mechanism."""

    repair_calls: int = 0
    calls_with_blocked_customers: int = 0
    blocked_customers: int = 0
    attempted_blocked_customers: int = 0
    successful_commits: int = 0
    successful_by_ejection_count: dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 0, 2: 0}
    )
    bounded_two_opt_trial_calls: int = 0
    successful_commits_using_bounded_two_opt: int = 0
    net_reward_gain_sum: float = 0.0
    module_seconds: float = 0.0
    new_best_candidates_with_successful_repair: int = 0
    successful_commits_in_new_best_candidates: int = 0
    success_cases: list[EjectionCaseRecord] = field(default_factory=list)
    failure_cases: list[EjectionCaseRecord] = field(default_factory=list)
    max_case_records_per_outcome: int = 12

    def record_case(self, case: EjectionCaseRecord) -> None:
        target = self.success_cases if case.outcome == "success" else self.failure_cases
        if len(target) < self.max_case_records_per_outcome:
            target.append(case)

    def mark_new_best_since(self, successful_commits_before: int) -> None:
        new_commits = self.successful_commits - successful_commits_before
        if new_commits <= 0:
            return
        self.new_best_candidates_with_successful_repair += 1
        self.successful_commits_in_new_best_candidates += new_commits
        for case in self.success_cases:
            if (
                case.commit_ordinal is not None
                and case.commit_ordinal > successful_commits_before
            ):
                case.produced_new_best = True

    def to_dict(self, total_solve_seconds: float | None = None) -> dict[str, object]:
        attempts = self.attempted_blocked_customers
        successes = self.successful_commits
        result: dict[str, object] = {
            "repair_calls": self.repair_calls,
            "calls_with_blocked_customers": self.calls_with_blocked_customers,
            "blocked_customers": self.blocked_customers,
            "attempted_blocked_customers": attempts,
            "successful_commits": successes,
            "success_rate_per_attempted_customer": (
                successes / attempts if attempts else 0.0
            ),
            "success_rate_per_repair_call": (
                successes / self.repair_calls if self.repair_calls else 0.0
            ),
            "successful_by_ejection_count": {
                str(key): value
                for key, value in sorted(self.successful_by_ejection_count.items())
            },
            "bounded_two_opt_trial_calls": self.bounded_two_opt_trial_calls,
            "successful_commits_using_bounded_two_opt": (
                self.successful_commits_using_bounded_two_opt
            ),
            "bounded_two_opt_share_of_successes": (
                self.successful_commits_using_bounded_two_opt / successes
                if successes
                else 0.0
            ),
            "mean_net_reward_gain_per_success": (
                self.net_reward_gain_sum / successes if successes else 0.0
            ),
            "net_reward_gain_sum": self.net_reward_gain_sum,
            "new_best_candidates_with_successful_repair": (
                self.new_best_candidates_with_successful_repair
            ),
            "successful_commits_in_new_best_candidates": (
                self.successful_commits_in_new_best_candidates
            ),
            "module_seconds": self.module_seconds,
            "success_cases": [asdict(case) for case in self.success_cases],
            "failure_cases": [asdict(case) for case in self.failure_cases],
        }
        if total_solve_seconds is not None:
            result["total_solve_seconds"] = total_solve_seconds
            result["module_runtime_share"] = (
                self.module_seconds / total_solve_seconds
                if total_solve_seconds > 0.0
                else 0.0
            )
        return result


def greedy_repair(
    partial_solution: TOPSolution,
    removed_nodes: list[int],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    rng: random.Random,
) -> TOPSolution:
    del rng
    result = partial_solution.copy()
    pending = build_repair_candidates(result, removed_nodes, instance)
    while pending:
        best: tuple[float, float, int, int, int] | None = None
        for node_id in sorted(pending):
            reward = instance.nodes[node_id].reward
            for route_index, route in enumerate(result.routes):
                current_distance = calculate_route_distance(
                    route, instance, distance_matrix
                )
                for position in range(1, len(route.node_ids)):
                    additional = insertion_cost(
                        route, position, node_id, distance_matrix
                    )
                    if current_distance + additional > instance.max_distance + 1e-9:
                        continue
                    score = float("inf") if additional <= 1e-12 else reward / additional
                    candidate = (score, reward, -node_id, -route_index, -position)
                    if best is None or candidate > best:
                        best = candidate
        if best is None:
            break
        _, _, negative_node, negative_route, negative_position = best
        node_id = -negative_node
        result.routes[-negative_route].node_ids.insert(-negative_position, node_id)
        pending.remove(node_id)
        result.visited_nodes.add(node_id)
    return update_solution_metrics(result, instance, distance_matrix)


def _try_ejection_insertion(
    node_id: int,
    result: TOPSolution,
    cache: AttentionRouteCache,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    max_positions: int = 3,
    max_ejections: int = 2,
    two_opt_passes: int | None = 5,
    ejectable_order_cache: dict[tuple[int, ...], list[int]] | None = None,
    removal_ranking: EjectionRemovalRanking = "removal_density",
    rng: random.Random | None = None,
    telemetry: EjectionTelemetry | None = None,
    iteration: int | None = None,
    stage: str = "repair",
) -> list[int] | None:
    """Force a high-value node in by ejecting low-value nodes, then 2-opt.

    Called only when no direct feasible insertion exists. The node is placed at
    one of the cheapest detour positions; the lowest reward-density nodes on
    that route are ejected jointly until the route fits after 2-opt. The move is
    kept only if it is feasible AND the net reward (inserted minus ejected)
    stays strictly positive, so the compound move never lowers total reward.

    On success the chosen route, ``result.visited_nodes`` and ``cache`` are
    updated in place, and the list of ejected node ids is returned. Otherwise
    the solution is untouched and ``None`` is returned.
    """
    # Import locally to avoid a repair <-> local_search import cycle.
    from .local_search import two_opt_route

    node_reward = instance.nodes[node_id].reward
    if telemetry is not None:
        telemetry.attempted_blocked_customers += 1
    if node_reward <= 0.0:
        if telemetry is not None:
            telemetry.record_case(
                EjectionCaseRecord(
                    "failure", iteration, stage, node_id, None, [], [], [], [],
                    None, None, None, None, instance.max_distance, None, [], 0,
                    False, False, None, "nonpositive_inserted_reward",
                )
            )
        return None

    forced_positions: list[tuple[float, int, int]] = []
    for route_index, route in enumerate(result.routes):
        for position in range(1, len(route.node_ids)):
            previous_id = route.node_ids[position - 1]
            next_id = route.node_ids[position]
            additional = (
                distance_matrix[previous_id][node_id]
                + distance_matrix[node_id][next_id]
                - distance_matrix[previous_id][next_id]
            )
            forced_positions.append((additional, route_index, position))
    forced_positions.sort(key=lambda item: item[0])

    best_key: tuple[float, float, int] | None = None
    best_payload: tuple[
        int, float, Route, list[int], EjectionCaseRecord | None
    ] | None = None
    closest_failure: EjectionCaseRecord | None = None
    saw_positive_gain_trial = False
    for additional, route_index, position in forced_positions[:max_positions]:
        route = result.routes[route_index]
        route_signature = tuple(route.node_ids)
        cached_ejectable = (
            ejectable_order_cache.get(route_signature)
            if ejectable_order_cache is not None
            else None
        )

        if cached_ejectable is None:
            interior = range(1, len(route.node_ids) - 1)

            def removal_density(index: int) -> float:
                previous_id = route.node_ids[index - 1]
                current_id = route.node_ids[index]
                next_id = route.node_ids[index + 1]
                saving = (
                    distance_matrix[previous_id][current_id]
                    + distance_matrix[current_id][next_id]
                    - distance_matrix[previous_id][next_id]
                )
                return instance.nodes[current_id].reward / max(
                    saving, 1e-12
                )

            if removal_ranking == "removal_density":
                ejectable_ids = [
                    route.node_ids[index]
                    for index in sorted(interior, key=removal_density)
                ]
            elif removal_ranking == "random":
                if rng is None:
                    raise ValueError("rng is required for random removal ranking")
                ejectable_ids = [route.node_ids[index] for index in interior]
                rng.shuffle(ejectable_ids)
            else:
                raise ValueError(f"unknown ejection removal ranking: {removal_ranking}")
            if ejectable_order_cache is not None:
                ejectable_order_cache[route_signature] = ejectable_ids
        else:
            ejectable_ids = cached_ejectable
        base_nodes = route.node_ids.copy()
        base_nodes.insert(position, node_id)
        original_distance = cache.route_distances[route_index]
        inserted_distance = original_distance + additional

        cap = min(max_ejections, len(ejectable_ids))
        for count in range(cap + 1):
            ejected = ejectable_ids[:count]
            ejected_reward = sum(
                instance.nodes[node].reward for node in ejected
            )
            net_reward = node_reward - ejected_reward
            if net_reward <= 1e-9:
                continue
            saw_positive_gain_trial = True
            ejected_set = set(ejected)
            trial_nodes = [
                node for node in base_nodes if node not in ejected_set
            ]
            # Cheap path first: check raw feasibility with no reordering. The
            # solver 2-opts every route right after repair, so a raw-feasible
            # move needs no in-trial 2-opt. Only pay for 2-opt to rescue a move
            # that the ejected savings alone cannot make feasible.
            trial_route = Route(trial_nodes)
            trial_distance = calculate_route_distance(
                trial_route, instance, distance_matrix
            )
            post_ejection_nodes = (
                trial_route.node_ids.copy() if telemetry is not None else []
            )
            post_ejection_distance = trial_distance
            two_opt_attempted = False
            two_opt_changed = False
            if trial_distance > instance.max_distance + 1e-9:
                two_opt_attempted = True
                if telemetry is not None:
                    telemetry.bounded_two_opt_trial_calls += 1
                trial_route = two_opt_route(
                    trial_route, instance, distance_matrix, two_opt_passes
                )
                two_opt_changed = (
                    telemetry is not None
                    and trial_route.node_ids != post_ejection_nodes
                )
                trial_distance = calculate_route_distance(
                    trial_route, instance, distance_matrix
                )
                if trial_distance > instance.max_distance + 1e-9:
                    failure = EjectionCaseRecord(
                        "failure", iteration, stage, node_id, route_index,
                        route.node_ids.copy(), base_nodes.copy(),
                        post_ejection_nodes, trial_route.node_ids.copy(),
                        original_distance, inserted_distance,
                        post_ejection_distance, trial_distance,
                        instance.max_distance,
                        max(0.0, inserted_distance - instance.max_distance),
                        ejected.copy(), count, two_opt_attempted,
                        two_opt_changed, net_reward,
                        "route_still_infeasible_after_bounded_two_opt",
                    )
                    if (
                        closest_failure is None
                        or (failure.final_distance or float("inf"))
                        < (closest_failure.final_distance or float("inf"))
                    ):
                        closest_failure = failure
                    continue
            # Compare on a scalar key only; Route objects are not orderable.
            key = (net_reward, -trial_distance, route_index)
            if best_key is None or key > best_key:
                best_key = key
                best_payload = (
                    route_index,
                    trial_distance,
                    trial_route,
                    ejected,
                    (
                        EjectionCaseRecord(
                            "success", iteration, stage, node_id, route_index,
                            route.node_ids.copy(), base_nodes.copy(),
                            post_ejection_nodes, trial_route.node_ids.copy(),
                            original_distance, inserted_distance,
                            post_ejection_distance, trial_distance,
                            instance.max_distance,
                            max(0.0, inserted_distance - instance.max_distance),
                            ejected.copy(), count, two_opt_attempted,
                            two_opt_changed, net_reward,
                        )
                        if telemetry is not None
                        else None
                    ),
                )
            # Fewest ejections that fit at this position is best; stop escalating.
            break

    if best_payload is None:
        if telemetry is not None:
            if closest_failure is None:
                closest_failure = EjectionCaseRecord(
                    "failure", iteration, stage, node_id, None, [], [], [], [],
                    None, None, None, None, instance.max_distance, None, [], 0,
                    False, False, None,
                    (
                        "no_positive_gain_ejection_prefix"
                        if not saw_positive_gain_trial
                        else "no_feasible_compound_move"
                    ),
                )
            telemetry.record_case(closest_failure)
        return None
    route_index, new_distance, new_route, ejected, success_case = best_payload
    result.routes[route_index] = new_route
    result.visited_nodes.add(node_id)
    for node in ejected:
        result.visited_nodes.discard(node)
    cache.set_route_distance(route_index, new_distance, instance)
    if telemetry is not None:
        assert success_case is not None
        telemetry.successful_commits += 1
        success_case.commit_ordinal = telemetry.successful_commits
        telemetry.successful_by_ejection_count[len(ejected)] = (
            telemetry.successful_by_ejection_count.get(len(ejected), 0) + 1
        )
        if success_case.bounded_two_opt_attempted:
            telemetry.successful_commits_using_bounded_two_opt += 1
        telemetry.net_reward_gain_sum += success_case.net_reward_gain or 0.0
        telemetry.record_case(success_case)
    return ejected


def attention_guided_repair(
    partial_solution: TOPSolution,
    removed_nodes: list[int],
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    rng: random.Random,
    weights: AttentionWeights | None = None,
    node_selection_strategy: NodeSelectionName = "dynamic_profit_time",
    candidate_pool_config: AttentionCandidatePoolConfig | None = None,
    enable_ejection: bool = False,
    ejection_max_positions: int = 3,
    ejection_max_nodes: int = 2,
    ejection_max_attempts: int = 6,
    ejection_two_opt_passes: int | None = 5,
    ejection_removal_ranking: EjectionRemovalRanking = "removal_density",
    ejection_telemetry: EjectionTelemetry | None = None,
    ejection_iteration: int | None = None,
    ejection_stage: str = "repair",
    *,
    use_ejection_cache: bool = True,
) -> TOPSolution:
    """Select nodes first, then place them using the Attention score.

    Nodes without a direct feasible insertion remain unvisited, unless
    ``enable_ejection`` is set: then a bounded compound "insert one high-value
    node, eject one or two low-value nodes, 2-opt" move is attempted so that
    reward-improving nodes can cross a tight route-distance barrier.

    ``ejection_max_attempts`` caps how many such compound moves are tried per
    repair call: the pending nodes that fail direct insertion are ranked by
    reward and only the most valuable few get an ejection attempt, since the
    reward gain comes from admitting a handful of high-value nodes, not from
    trying every candidate.
    """
    result = partial_solution.copy()
    if enable_ejection and ejection_telemetry is not None:
        ejection_telemetry.repair_calls += 1
    cache = AttentionRouteCache.from_solution(
        result, instance, distance_matrix
    )
    pending = build_attention_repair_candidates(
        result,
        removed_nodes,
        instance,
        distance_matrix,
        cache.route_distances,
        rng,
        candidate_pool_config,
    )
    effective_weights = weights or AttentionWeights()
    best_additional_profile = build_best_additional_profile(
        pending,
        result,
        cache.route_distances,
        instance,
        distance_matrix,
    )

    blocked_nodes: list[int] = []
    while pending:
        node_id = select_next_node(
            node_selection_strategy,
            pending,
            removed_nodes,
            result,
            cache.route_distances,
            instance,
            distance_matrix,
            rng,
            best_additional_profile,
        )
        pending.remove(node_id)
        best: tuple[float, int, int] | None = None
        best_additional_distance = 0.0
        candidates: list[
            tuple[int, int, float, AttentionScoreComponents]
        ] = []
        for route_index, route in enumerate(result.routes):
            for position in range(1, len(route.node_ids)):
                previous_id = route.node_ids[position - 1]
                next_id = route.node_ids[position]
                previous_distance = distance_matrix[previous_id][node_id]
                next_distance = distance_matrix[node_id][next_id]
                additional_distance = (
                    previous_distance
                    + next_distance
                    - distance_matrix[previous_id][next_id]
                )
                new_route_distance = (
                    cache.route_distances[route_index]
                    + additional_distance
                )
                if new_route_distance > instance.max_distance + 1e-9:
                    continue
                components = incremental_attention_score_components(
                    reward=instance.nodes[node_id].reward,
                    additional_distance=additional_distance,
                    new_route_distance=new_route_distance,
                    mean_neighbor_distance=(
                        previous_distance + next_distance
                    )
                    / 2.0,
                    route_index=route_index,
                    cache=cache,
                    instance=instance,
                )
                candidates.append(
                    (
                        route_index,
                        position,
                        additional_distance,
                        components,
                    )
                )

        normalized_components = normalize_attention_components(
            [candidate[3] for candidate in candidates]
        )
        for candidate_data, normalized in zip(
            candidates, normalized_components
        ):
            (
                route_index,
                position,
                additional_distance,
                _,
            ) = candidate_data
            score = normalized.weighted_sum(effective_weights)
            candidate = (
                score,
                -route_index,
                -position,
            )
            if best is None or candidate > best:
                best = candidate
                best_additional_distance = additional_distance
        if best is None:
            if enable_ejection:
                blocked_nodes.append(node_id)
            continue
        _, negative_route, negative_position = best
        route_index = -negative_route
        result.routes[route_index].node_ids.insert(
            -negative_position, node_id
        )
        cache.apply_insertion(
            route_index, best_additional_distance, instance
        )
        result.visited_nodes.add(node_id)

    # After every directly feasible insertion, spend a bounded ejection budget
    # on the highest-reward nodes that were blocked by the route-distance wall.
    if enable_ejection and blocked_nodes:
        started = perf_counter() if ejection_telemetry is not None else None
        if ejection_telemetry is not None:
            ejection_telemetry.calls_with_blocked_customers += 1
            ejection_telemetry.blocked_customers += len(blocked_nodes)
        ejectable_order_cache: dict[tuple[int, ...], list[int]] | None = (
            {} if use_ejection_cache else None
        )
        blocked_nodes.sort(
            key=lambda node: instance.nodes[node].reward, reverse=True
        )
        for node_id in blocked_nodes[:ejection_max_attempts]:
            _try_ejection_insertion(
                node_id,
                result,
                cache,
                instance,
                distance_matrix,
                ejection_max_positions,
                ejection_max_nodes,
                ejection_two_opt_passes,
                ejectable_order_cache,
                ejection_removal_ranking,
                rng,
                ejection_telemetry,
                ejection_iteration,
                ejection_stage,
            )
        if ejection_telemetry is not None and started is not None:
            ejection_telemetry.module_seconds += perf_counter() - started

    return update_solution_metrics(result, instance, distance_matrix)
