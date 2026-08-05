from pathlib import Path
from itertools import permutations

import pytest

from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.kim_alns import KimALNSConfig, solve_kim_alns
from top_alns.kim_alns.solver import _best_pair_insertion
from top_alns.parser import parse_instance


SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_kim_alns_is_feasible_reproducible_and_counts_outer_iterations() -> None:
    instance = parse_instance(SAMPLE)
    first = solve_kim_alns(
        instance,
        max_iterations=5,
        seed=7,
        config=KimALNSConfig(random_replacement_iterations=3),
    )
    second = solve_kim_alns(
        instance,
        max_iterations=5,
        seed=7,
        config=KimALNSConfig(random_replacement_iterations=3),
    )
    assert first.stats.completed_iterations == 5
    assert first.stats.termination_reason == "fixed_iterations"
    assert check_solution_feasible(
        first.solution, instance, build_distance_matrix(instance)
    )
    assert [route.node_ids for route in first.solution.routes] == [
        route.node_ids for route in second.solution.routes
    ]


def test_kim_alns_hard_wall_returns_a_feasible_solution() -> None:
    instance = parse_instance(SAMPLE)
    result = solve_kim_alns(
        instance,
        max_iterations=1_000_000,
        seed=3,
        time_limit_seconds=0.001,
        config=KimALNSConfig(random_replacement_iterations=3),
    )
    assert result.stats.timed_out
    assert result.stats.runtime_seconds < 0.20
    assert check_solution_feasible(
        result.solution, instance, build_distance_matrix(instance)
    )


def test_kim_alns_rejects_invalid_configuration() -> None:
    instance = parse_instance(SAMPLE)
    with pytest.raises(ValueError, match="pool_size"):
        solve_kim_alns(instance, config=KimALNSConfig(pool_size=0))


def test_pair_insertion_matches_exhaustive_enumeration() -> None:
    instance = parse_instance(SAMPLE)
    matrix = build_distance_matrix(instance)
    customers = sorted(set(instance.nodes) - instance.depot_ids)
    base_customer, first, second = customers[:3]
    base = [instance.depot_id, base_customer, instance.route_end_id]
    base_distance = sum(matrix[a][b] for a, b in zip(base, base[1:]))
    extra, inserted = _best_pair_insertion(base, first, second, matrix)
    candidate_routes = [
        [instance.depot_id, *order, instance.route_end_id]
        for order in permutations((base_customer, first, second))
    ]
    exhaustive = min(
        candidate_routes,
        key=lambda nodes: sum(matrix[a][b] for a, b in zip(nodes, nodes[1:])),
    )
    exhaustive_distance = sum(
        matrix[a][b] for a, b in zip(exhaustive, exhaustive[1:])
    )
    inserted_distance = sum(
        matrix[a][b] for a, b in zip(inserted, inserted[1:])
    )
    assert base_distance + extra == pytest.approx(inserted_distance)
    assert inserted_distance == pytest.approx(exhaustive_distance)
