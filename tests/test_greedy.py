from pathlib import Path

from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.greedy import greedy_initial_solution
from top_alns.parser import parse_instance

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_greedy_returns_feasible_non_negative_solution() -> None:
    instance = parse_instance(SAMPLE)
    matrix = build_distance_matrix(instance)
    solution = greedy_initial_solution(instance, matrix)
    assert check_solution_feasible(solution, instance, matrix)
    assert solution.total_reward >= 0
