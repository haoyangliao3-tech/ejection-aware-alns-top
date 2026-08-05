"""A minimal, reproducible ALNS solver."""

from __future__ import annotations

from collections.abc import Callable
import math
import random
from math import log
from time import perf_counter

from ..distance import DistanceMatrix, build_distance_matrix
from ..feasibility import check_solution_feasible
from ..greedy import greedy_initial_solution
from ..models import TOPInstance, TOPSolution
from ..utils.random import create_rng
from .acceptance import (
    SimulatedAnnealingAcceptance,
    geometric_cooling_rate,
)
from .candidates import AttentionCandidatePoolConfig
from .destroy import (
    largest_saving_removal,
    low_reward_density_removal,
    random_removal,
    route_removal,
    sequence_removal,
)
from .exchange import prize_collecting_exchange
from .attention import AttentionWeights
from .local_search import improve_then_attention_residual_repair
from .operator_selection import AdaptiveOperatorSelector
from .repair import (
    EjectionRemovalRanking,
    EjectionTelemetry,
    attention_guided_repair,
)
from .node_selection import NodeSelectionName

DestroyOperator = Callable[
    [TOPSolution, TOPInstance, DistanceMatrix, int, random.Random],
    tuple[TOPSolution, list[int]],
]

WALL_CLOCK_ANNEALING_STEPS = 1_000_000


def _is_better(candidate: TOPSolution, incumbent: TOPSolution) -> bool:
    if candidate.total_reward > incumbent.total_reward + 1e-9:
        return True
    return (
        abs(candidate.total_reward - incumbent.total_reward) <= 1e-9
        and candidate.total_distance < incumbent.total_distance - 1e-9
    )


def _random_destroy(
    solution: TOPSolution,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    remove_count: int,
    rng: random.Random,
) -> tuple[TOPSolution, list[int]]:
    del instance, distance_matrix
    return random_removal(solution, remove_count, rng)


class ALNSolver:
    def __init__(
        self,
        max_iterations: int = 100,
        removal_rate: float = 0.2,
        minimum_removal_rate: float = 0.05,
        random_seed: int | None = 0,
        attention_weights: AttentionWeights | None = None,
        initial_temperature: float = 20.0,
        cooling_rate: float | None = None,
        minimum_temperature: float = 0.01,
        candidate_pool_config: AttentionCandidatePoolConfig | None = None,
        operator_segment_size: int = 100,
        exchange_stagnation: int = 200,
        exchange_top_unvisited: int = 20,
        exchange_positions_per_node: int = 3,
        exchange_ejection_pool_size: int = 15,
        exchange_max_ejections: int = 2,
        reheating_factor: float = 0.25,
        enable_ejection: bool = True,
        ejection_max_positions: int = 3,
        ejection_max_nodes: int = 2,
        ejection_max_attempts: int = 100,
        ejection_two_opt_passes: int | None = 1,
        ejection_removal_ranking: EjectionRemovalRanking = "removal_density",
        collect_ejection_telemetry: bool = False,
        sa_reward_scaled: bool = False,
        sa_reward_scale_factor: float = 1.0,
        sa_cyclic_reheat: bool = False,
        sa_reheat_cycles: int = 5,
        sa_wall_clock_horizon_seconds: float | None = None,
    ) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not 0.0 < removal_rate <= 1.0:
            raise ValueError("removal_rate must be in (0, 1]")
        if not 0.0 < minimum_removal_rate <= removal_rate:
            raise ValueError(
                "minimum_removal_rate must be in (0, removal_rate]"
            )
        if operator_segment_size <= 0:
            raise ValueError("operator_segment_size must be positive")
        if exchange_stagnation < 0:
            raise ValueError("exchange_stagnation must be non-negative")
        if not 0.0 < reheating_factor <= 1.0:
            raise ValueError("reheating_factor must be in (0, 1]")
        if exchange_top_unvisited < 0:
            raise ValueError("exchange_top_unvisited must be non-negative")
        if exchange_positions_per_node < 0:
            raise ValueError(
                "exchange_positions_per_node must be non-negative"
            )
        if exchange_ejection_pool_size < 0:
            raise ValueError(
                "exchange_ejection_pool_size must be non-negative"
            )
        if exchange_max_ejections < 0:
            raise ValueError("exchange_max_ejections must be non-negative")
        if sa_reward_scale_factor <= 0.0:
            raise ValueError("sa_reward_scale_factor must be positive")
        if sa_reheat_cycles <= 0:
            raise ValueError("sa_reheat_cycles must be positive")
        if (
            sa_wall_clock_horizon_seconds is not None
            and sa_wall_clock_horizon_seconds <= 0.0
        ):
            raise ValueError("sa_wall_clock_horizon_seconds must be positive")
        if sa_wall_clock_horizon_seconds is not None and sa_cyclic_reheat:
            raise ValueError(
                "wall-clock cooling and cyclic reheating cannot be combined"
            )
        if ejection_removal_ranking not in {"removal_density", "random"}:
            raise ValueError(
                "ejection_removal_ranking must be 'removal_density' or 'random'"
            )
        self.max_iterations = max_iterations
        self.removal_rate = removal_rate
        self.minimum_removal_rate = minimum_removal_rate
        self.random_seed = random_seed
        self.attention_weights = attention_weights or AttentionWeights()
        self.candidate_pool_config = candidate_pool_config
        self.operator_segment_size = operator_segment_size
        self.exchange_stagnation = exchange_stagnation
        self.exchange_top_unvisited = exchange_top_unvisited
        self.exchange_positions_per_node = exchange_positions_per_node
        self.exchange_ejection_pool_size = exchange_ejection_pool_size
        self.exchange_max_ejections = exchange_max_ejections
        self.reheating_factor = reheating_factor
        self.enable_ejection = enable_ejection
        self.ejection_max_positions = ejection_max_positions
        self.ejection_max_nodes = ejection_max_nodes
        self.ejection_max_attempts = ejection_max_attempts
        self.ejection_two_opt_passes = ejection_two_opt_passes
        self.ejection_removal_ranking = ejection_removal_ranking
        self.collect_ejection_telemetry = collect_ejection_telemetry
        self.sa_reward_scaled = sa_reward_scaled
        self.sa_reward_scale_factor = sa_reward_scale_factor
        self.sa_cyclic_reheat = sa_cyclic_reheat
        self.sa_reheat_cycles = sa_reheat_cycles
        self.sa_wall_clock_horizon_seconds = sa_wall_clock_horizon_seconds
        # Raw schedule inputs kept so solve() can recompute the temperature
        # schedule once the instance (hence the reward scale) is known.
        self.initial_temperature = initial_temperature
        self.minimum_temperature = minimum_temperature
        self.cooling_rate = cooling_rate
        effective_cooling_rate = (
            geometric_cooling_rate(
                initial_temperature,
                minimum_temperature,
                max_iterations,
            )
            if cooling_rate is None
            else cooling_rate
        )
        self.acceptance = SimulatedAnnealingAcceptance(
            initial_temperature=initial_temperature,
            cooling_rate=effective_cooling_rate,
            minimum_temperature=minimum_temperature,
        )
        self.last_run_stats: dict[str, object] = {}

    def _build_schedule(
        self, instance: TOPInstance
    ) -> tuple[SimulatedAnnealingAcceptance, int | None]:
        """Resolve the acceptance schedule and reheat period for a run.

        When neither SA enhancement is enabled the configured ``self.acceptance``
        is returned unchanged, so the ablation ``--no`` paths reproduce the
        original geometric-cooling behaviour exactly. ``sa_reward_scaled`` sets
        the initial temperature to a multiple of the median node reward so that
        losing a typical node is genuinely acceptable early on; ``sa_cyclic_
        reheat`` calibrates cooling to a single cycle so the temperature can be
        reset periodically instead of decaying once to ``Tmin``.
        """
        if not self.sa_reward_scaled and not self.sa_cyclic_reheat:
            return self.acceptance, None

        initial_temperature = self.initial_temperature
        if self.sa_reward_scaled:
            rewards = sorted(
                instance.nodes[node_id].reward
                for node_id in instance.nodes
                if node_id not in instance.depot_ids
            )
            if rewards:
                count = len(rewards)
                median_reward = (
                    rewards[count // 2]
                    if count % 2
                    else (rewards[count // 2 - 1] + rewards[count // 2]) / 2.0
                )
                scaled = self.sa_reward_scale_factor * median_reward
                initial_temperature = max(scaled, self.minimum_temperature)

        reheat_period: int | None = None
        horizon = self.max_iterations
        if self.sa_cyclic_reheat and self.max_iterations > 0:
            reheat_period = max(
                1, self.max_iterations // self.sa_reheat_cycles
            )
            horizon = reheat_period

        cooling_rate = (
            geometric_cooling_rate(
                initial_temperature, self.minimum_temperature, horizon
            )
            if self.cooling_rate is None or self.sa_cyclic_reheat
            else self.cooling_rate
        )
        acceptance = SimulatedAnnealingAcceptance(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            minimum_temperature=self.minimum_temperature,
        )
        return acceptance, reheat_period

    def solve(
        self,
        instance: TOPInstance,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TOPSolution:
        solve_started = perf_counter()
        rng = create_rng(self.random_seed)
        distance_matrix = build_distance_matrix(instance)
        acceptance, reheat_period = self._build_schedule(instance)
        if self.sa_wall_clock_horizon_seconds is not None:
            acceptance = SimulatedAnnealingAcceptance(
                initial_temperature=acceptance.initial_temperature,
                cooling_rate=geometric_cooling_rate(
                    acceptance.initial_temperature,
                    acceptance.minimum_temperature,
                    WALL_CLOCK_ANNEALING_STEPS,
                ),
                minimum_temperature=acceptance.minimum_temperature,
            )
        current = greedy_initial_solution(instance, distance_matrix)
        if not check_solution_feasible(current, instance, distance_matrix):
            raise RuntimeError("greedy constructor produced an infeasible solution")
        best = current.copy()
        initial_reward = current.total_reward
        selector: AdaptiveOperatorSelector[DestroyOperator] = (
            AdaptiveOperatorSelector(
                {
                    "random": _random_destroy,
                    "low_reward_density": low_reward_density_removal,
                    "largest_saving": largest_saving_removal,
                    "sequence": sequence_removal,
                    "route": route_removal,
                }
            )
        )
        node_selector: AdaptiveOperatorSelector[NodeSelectionName] = (
            AdaptiveOperatorSelector(
                {
                    "dynamic_profit_time": "dynamic_profit_time",
                    "highest_profit": "highest_profit",
                    "random": "random",
                    "lrfi": "lrfi",
                }
            )
        )

        if progress_callback is not None:
            progress_callback(0, self.max_iterations)
        completed_iterations = 0
        iterations_since_best = 0
        annealing_age = 0
        wall_clock_epoch_started = solve_started
        wall_clock_epoch_age = 0

        def current_annealing_age() -> int:
            if self.sa_wall_clock_horizon_seconds is None:
                return annealing_age
            elapsed = max(0.0, perf_counter() - wall_clock_epoch_started)
            elapsed_fraction = min(
                1.0,
                elapsed / self.sa_wall_clock_horizon_seconds,
            )
            return min(
                WALL_CLOCK_ANNEALING_STEPS,
                wall_clock_epoch_age
                + round(WALL_CLOCK_ANNEALING_STEPS * elapsed_fraction),
            )
        best_improvements = 0
        last_improvement_iteration = 0
        exchange_attempts = 0
        exchange_successes = 0
        reheats = 0
        ejection_telemetry = (
            EjectionTelemetry()
            if self.enable_ejection and self.collect_ejection_telemetry
            else None
        )
        for iteration in range(self.max_iterations):
            if not current.visited_nodes:
                break
            stagnation_ratio = (
                0.0
                if self.exchange_stagnation <= 0
                else min(
                    1.0,
                    iterations_since_best / self.exchange_stagnation,
                )
            )
            lower_rate = self.minimum_removal_rate + (
                self.removal_rate - self.minimum_removal_rate
            ) * 0.5 * stagnation_ratio
            effective_removal_rate = rng.uniform(
                lower_rate, self.removal_rate
            )
            remove_count = max(
                1,
                math.ceil(
                    len(current.visited_nodes) * effective_removal_rate
                ),
            )
            operator_name, destroy = selector.select(rng)
            node_strategy_name, node_strategy = node_selector.select(rng)
            partial, removed = destroy(
                current, instance, distance_matrix, remove_count, rng
            )
            commits_before_candidate = (
                ejection_telemetry.successful_commits
                if ejection_telemetry is not None
                else 0
            )
            candidate = attention_guided_repair(
                partial,
                removed,
                instance,
                distance_matrix,
                rng,
                self.attention_weights,
                node_selection_strategy=node_strategy,
                candidate_pool_config=self.candidate_pool_config,
                enable_ejection=self.enable_ejection,
                ejection_max_positions=self.ejection_max_positions,
                ejection_max_nodes=self.ejection_max_nodes,
                ejection_max_attempts=self.ejection_max_attempts,
                ejection_two_opt_passes=self.ejection_two_opt_passes,
                ejection_removal_ranking=self.ejection_removal_ranking,
                ejection_telemetry=ejection_telemetry,
                ejection_iteration=iteration + 1,
                ejection_stage="main_repair",
            )
            candidate = improve_then_attention_residual_repair(
                candidate,
                instance,
                distance_matrix,
                rng,
                self.attention_weights,
                node_strategy,
                remove_count,
                self.candidate_pool_config,
                enable_ejection=self.enable_ejection,
                ejection_max_positions=self.ejection_max_positions,
                ejection_max_nodes=self.ejection_max_nodes,
                ejection_max_attempts=self.ejection_max_attempts,
                ejection_two_opt_passes=self.ejection_two_opt_passes,
                ejection_removal_ranking=self.ejection_removal_ranking,
                ejection_telemetry=ejection_telemetry,
                ejection_iteration=iteration + 1,
                ejection_stage="residual_repair",
            )
            accepted = (
                check_solution_feasible(candidate, instance, distance_matrix)
                and acceptance.accept(
                    current.total_reward,
                    candidate.total_reward,
                    current_annealing_age(),
                    rng,
                    current.total_distance,
                    candidate.total_distance,
                )
            )
            found_new_best = False
            if accepted:
                current = candidate
                if _is_better(candidate, best):
                    best = candidate.copy()
                    found_new_best = True
                    best_improvements += 1
                    last_improvement_iteration = iteration + 1
                    if ejection_telemetry is not None:
                        ejection_telemetry.mark_new_best_since(
                            commits_before_candidate
                        )
                    selector.update(operator_name, 5.0)
                    node_selector.update(node_strategy_name, 5.0)
                else:
                    selector.update(operator_name, 1.0)
                    node_selector.update(node_strategy_name, 1.0)
            else:
                selector.update(operator_name, 0.0)
                node_selector.update(node_strategy_name, 0.0)
            if found_new_best:
                iterations_since_best = 0
            else:
                iterations_since_best += 1
            annealing_age += 1

            should_exchange = (
                self.exchange_stagnation > 0
                and iterations_since_best >= self.exchange_stagnation
                and iterations_since_best % self.exchange_stagnation == 0
            )
            if should_exchange:
                commits_before_exchange = (
                    ejection_telemetry.successful_commits
                    if ejection_telemetry is not None
                    else 0
                )
                exchange_attempts += 1
                exchange = prize_collecting_exchange(
                    best,
                    instance,
                    distance_matrix,
                    top_unvisited=self.exchange_top_unvisited,
                    positions_per_node=self.exchange_positions_per_node,
                    ejection_pool_size=self.exchange_ejection_pool_size,
                    max_ejections=self.exchange_max_ejections,
                )
                exchanged = exchange.solution
                if _is_better(exchanged, best):
                    exchanged = attention_guided_repair(
                        exchanged,
                        exchange.ejected_nodes,
                        instance,
                        distance_matrix,
                        rng,
                        self.attention_weights,
                        node_selection_strategy=node_strategy,
                        candidate_pool_config=self.candidate_pool_config,
                        enable_ejection=self.enable_ejection,
                        ejection_max_positions=self.ejection_max_positions,
                        ejection_max_nodes=self.ejection_max_nodes,
                        ejection_max_attempts=self.ejection_max_attempts,
                        ejection_two_opt_passes=self.ejection_two_opt_passes,
                        ejection_removal_ranking=self.ejection_removal_ranking,
                        ejection_telemetry=ejection_telemetry,
                        ejection_iteration=iteration + 1,
                        ejection_stage="exchange_repair",
                    )
                    exchanged = improve_then_attention_residual_repair(
                        exchanged,
                        instance,
                        distance_matrix,
                        rng,
                        self.attention_weights,
                        node_strategy,
                        remove_count,
                        self.candidate_pool_config,
                        enable_ejection=self.enable_ejection,
                        ejection_max_positions=self.ejection_max_positions,
                        ejection_max_nodes=self.ejection_max_nodes,
                        ejection_max_attempts=self.ejection_max_attempts,
                        ejection_two_opt_passes=self.ejection_two_opt_passes,
                        ejection_removal_ranking=self.ejection_removal_ranking,
                        ejection_telemetry=ejection_telemetry,
                        ejection_iteration=iteration + 1,
                        ejection_stage="exchange_residual_repair",
                    )
                if (
                    check_solution_feasible(
                        exchanged, instance, distance_matrix
                    )
                    and _is_better(exchanged, best)
                ):
                    best = exchanged.copy()
                    current = exchanged
                    iterations_since_best = 0
                    exchange_successes += 1
                    best_improvements += 1
                    last_improvement_iteration = iteration + 1
                    if ejection_telemetry is not None:
                        ejection_telemetry.mark_new_best_since(
                            commits_before_exchange
                        )
                else:
                    reheats += 1
                    if self.sa_wall_clock_horizon_seconds is not None:
                        wall_clock_epoch_started = perf_counter()
                        wall_clock_epoch_age = (
                            max(
                                0,
                                round(
                                    log(self.reheating_factor)
                                    / log(acceptance.cooling_rate)
                                ),
                            )
                            if acceptance.cooling_rate < 1.0
                            else 0
                        )
                    elif acceptance.cooling_rate < 1.0:
                        annealing_age = max(
                            0,
                            round(
                                log(self.reheating_factor)
                                / log(acceptance.cooling_rate)
                            ),
                        )
                    else:
                        annealing_age = 0

            # Cyclic (sawtooth) reheating: reset the annealing clock on a fixed
            # schedule so the temperature climbs back to T0 instead of staying
            # glued to Tmin, keeping the ability to escape local optima all run.
            if (
                reheat_period is not None
                and (iteration + 1) % reheat_period == 0
            ):
                annealing_age = 0
                reheats += 1

            if (iteration + 1) % self.operator_segment_size == 0:
                selector.end_segment()
                node_selector.end_segment()
            if progress_callback is not None:
                progress_callback(iteration + 1, self.max_iterations)
            completed_iterations = iteration + 1
        if (
            progress_callback is not None
            and completed_iterations < self.max_iterations
        ):
            progress_callback(self.max_iterations, self.max_iterations)
        total_solve_seconds = perf_counter() - solve_started
        self.last_run_stats = {
            "initial_reward": initial_reward,
            "completed_iterations": completed_iterations,
            "best_improvements": best_improvements,
            "last_improvement_iteration": last_improvement_iteration,
            "iterations_since_best": iterations_since_best,
            "exchange_attempts": exchange_attempts,
            "exchange_successes": exchange_successes,
            "reheats": reheats,
            "final_temperature": acceptance.temperature(
                current_annealing_age()
            ),
            "sa_initial_temperature": acceptance.initial_temperature,
            "sa_cooling_rate": acceptance.cooling_rate,
            "sa_reheat_period": reheat_period,
            "sa_schedule": (
                "wall_clock_geometric"
                if self.sa_wall_clock_horizon_seconds is not None
                else "iteration_geometric"
            ),
            "sa_wall_clock_horizon_seconds": (
                self.sa_wall_clock_horizon_seconds
            ),
            "destroy_weights": {
                name: score.weight for name, score in selector.scores.items()
            },
            "node_selection_weights": {
                name: score.weight
                for name, score in node_selector.scores.items()
            },
            "ejection_configuration": {
                "enabled": self.enable_ejection,
                "max_positions": self.ejection_max_positions,
                "max_nodes": self.ejection_max_nodes,
                "max_attempts": self.ejection_max_attempts,
                "two_opt_passes": self.ejection_two_opt_passes,
                "removal_ranking": self.ejection_removal_ranking,
            },
        }
        if ejection_telemetry is not None:
            self.last_run_stats["ejection_telemetry"] = (
                ejection_telemetry.to_dict(total_solve_seconds)
            )
        return best
