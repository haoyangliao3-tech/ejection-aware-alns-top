import random

from top_alns.alns.attention import AttentionWeights
from top_alns.alns.local_search import (
    improve_then_attention_residual_repair,
)
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.models import Node, Route, TOPInstance, TOPSolution
from top_alns.solution import update_solution_metrics


def test_two_opt_released_distance_is_used_by_residual_repair() -> None:
    instance = TOPInstance(
        nodes={
            0: Node(0, 0.0, 0.0, 0.0),
            1: Node(1, 0.0, 10.0, 10.0),
            2: Node(2, 10.0, 0.0, 10.0),
            3: Node(3, 10.0, 10.0, 10.0),
            4: Node(4, 5.0, 10.0, 50.0),
        },
        depot_id=0,
        vehicle_count=1,
        max_distance=49.0,
    )
    matrix = build_distance_matrix(instance)
    crossing = update_solution_metrics(
        TOPSolution([Route([0, 1, 2, 3, 0])]),
        instance,
        matrix,
    )

    result = improve_then_attention_residual_repair(
        crossing,
        instance,
        matrix,
        random.Random(0),
        AttentionWeights(),
        "dynamic_profit_time",
        residual_bucket_size=1,
    )

    assert result.total_distance < crossing.total_distance
    assert result.visited_nodes == {1, 2, 3, 4}
    assert check_solution_feasible(result, instance, matrix)
