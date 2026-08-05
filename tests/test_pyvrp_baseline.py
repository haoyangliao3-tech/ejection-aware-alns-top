from pathlib import Path

import pytest

pytest.importorskip("pyvrp")

from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.open_source_baselines import solve_pyvrp_top
from top_alns.parser import parse_instance


SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_pyvrp_top_returns_a_feasible_solution() -> None:
    instance = parse_instance(SAMPLE)
    solution, _ = solve_pyvrp_top(instance, seed=7, time_limit_seconds=0.05)
    assert check_solution_feasible(
        solution, instance, build_distance_matrix(instance)
    )
