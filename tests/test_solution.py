from pathlib import Path
import random

from top_alns.alns.destroy import random_removal
from top_alns.alns.repair import greedy_repair
from top_alns.alns.solver import ALNSolver
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.models import Route, TOPSolution
from top_alns.parser import parse_instance
from top_alns.solution import calculate_solution_reward, get_visited_nodes
from top_alns.greedy import greedy_initial_solution

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_reward_counts_unique_non_depot_nodes() -> None:
    instance = parse_instance(SAMPLE)
    solution = TOPSolution([Route([0, 1, 0]), Route([0, 1, 2, 0])])
    assert get_visited_nodes(solution) == {0, 1, 2}
    assert calculate_solution_reward(solution, instance) == 50


def test_solver_runs_and_returns_feasible_solution() -> None:
    instance = parse_instance(SAMPLE)
    solution = ALNSolver(max_iterations=10, random_seed=7).solve(instance)
    matrix = build_distance_matrix(instance)
    assert check_solution_feasible(solution, instance, matrix)


def test_destroy_repair_round_trip_keeps_metrics_consistent() -> None:
    instance = parse_instance(SAMPLE)
    matrix = build_distance_matrix(instance)
    original = greedy_initial_solution(instance, matrix)
    partial, removed = random_removal(original, 1, random.Random(1))
    assert set(removed).isdisjoint(get_visited_nodes(partial))
    repaired = greedy_repair(
        partial, removed, instance, matrix, random.Random(1)
    )
    assert repaired.total_reward == calculate_solution_reward(repaired, instance)
    assert check_solution_feasible(repaired, instance, matrix)


def test_attention_solver_is_reproducible() -> None:
    instance = parse_instance(SAMPLE)
    first = ALNSolver(max_iterations=10, random_seed=11).solve(instance)
    second = ALNSolver(max_iterations=10, random_seed=11).solve(instance)
    assert [route.node_ids for route in first.routes] == [
        route.node_ids for route in second.routes
    ]
