from pathlib import Path

import pytest

pytest.importorskip("pyvrp")

from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.models import Node, TOPInstance
from top_alns.open_source_baselines import solve_pyvrp_top
from top_alns.parser import parse_instance


SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_pyvrp_top_returns_a_feasible_solution() -> None:
    instance = parse_instance(SAMPLE)
    solution, _ = solve_pyvrp_top(instance, seed=7, time_limit_seconds=0.05)
    assert check_solution_feasible(
        solution, instance, build_distance_matrix(instance)
    )


def test_pyvrp_top_supports_distinct_start_and_end_depots() -> None:
    instance = TOPInstance(
        nodes={
            0: Node(0, 0.0, 0.0, 0.0),
            1: Node(1, 1.0, 0.0, 10.0),
            2: Node(2, 2.0, 0.0, 10.0),
            3: Node(3, 3.0, 0.0, 0.0),
        },
        depot_id=0,
        vehicle_count=1,
        max_distance=4.0,
        end_depot_id=3,
    )

    solution, _ = solve_pyvrp_top(
        instance,
        seed=7,
        time_limit_seconds=0.2,
    )

    assert solution.routes[0].node_ids[0] == 0
    assert solution.routes[0].node_ids[-1] == 3
    assert solution.total_reward > 0
    assert check_solution_feasible(
        solution, instance, build_distance_matrix(instance)
    )
