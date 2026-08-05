from pathlib import Path

import pytest

from top_alns.comparison_baselines import solve_comparison_baseline
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.parser import parse_instance


SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


@pytest.mark.parametrize("algorithm", ["grasp", "ils", "vns"])
def test_comparison_baseline_is_feasible_and_reproducible(algorithm: str) -> None:
    instance = parse_instance(SAMPLE)
    matrix = build_distance_matrix(instance)
    first = solve_comparison_baseline(
        instance, algorithm, max_iterations=30, seed=7
    )
    second = solve_comparison_baseline(
        instance, algorithm, max_iterations=30, seed=7
    )
    assert check_solution_feasible(first, instance, matrix)
    assert [route.node_ids for route in first.routes] == [
        route.node_ids for route in second.routes
    ]
    assert first.total_reward == second.total_reward


@pytest.mark.parametrize("algorithm", ["grasp", "ils", "vns"])
def test_time_limit_returns_a_feasible_solution(algorithm: str) -> None:
    instance = parse_instance(SAMPLE)
    matrix = build_distance_matrix(instance)
    solution = solve_comparison_baseline(
        instance,
        algorithm,
        max_iterations=1_000_000,
        seed=7,
        time_limit_seconds=0.01,
    )
    assert check_solution_feasible(solution, instance, matrix)
