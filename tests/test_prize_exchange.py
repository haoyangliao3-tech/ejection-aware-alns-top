from top_alns.alns.exchange import prize_collecting_exchange
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.models import Node, Route, TOPInstance, TOPSolution
from top_alns.solution import update_solution_metrics


def test_exchange_can_insert_profitable_node_by_ejecting_one() -> None:
    instance = TOPInstance(
        nodes={
            0: Node(0, 0.0, 0.0, 0.0),
            1: Node(1, 5.0, 0.0, 10.0),
            2: Node(2, 10.0, 0.0, 10.0),
            3: Node(3, 5.0, 5.0, 100.0),
        },
        depot_id=0,
        vehicle_count=1,
        max_distance=20.0,
    )
    matrix = build_distance_matrix(instance)
    current = update_solution_metrics(
        TOPSolution([Route([0, 1, 2, 0])]),
        instance,
        matrix,
    )

    result = prize_collecting_exchange(
        current,
        instance,
        matrix,
        top_unvisited=1,
        positions_per_node=3,
        ejection_pool_size=2,
        max_ejections=1,
    )

    assert result.attempted
    assert result.solution.total_reward > current.total_reward
    assert 3 in result.solution.visited_nodes
    assert len(result.ejected_nodes) == 1
    assert check_solution_feasible(result.solution, instance, matrix)
