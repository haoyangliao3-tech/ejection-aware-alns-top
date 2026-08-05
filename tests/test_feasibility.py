from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_route_feasible, validate_solution
from top_alns.models import Node, Route, TOPInstance, TOPSolution


def make_instance(max_distance: float = 10.0) -> TOPInstance:
    return TOPInstance(
        nodes={0: Node(0, 0, 0, 0), 1: Node(1, 3, 0, 5)},
        depot_id=0,
        vehicle_count=2,
        max_distance=max_distance,
    )


def test_empty_depot_route_is_feasible() -> None:
    instance = make_instance()
    matrix = build_distance_matrix(instance)
    assert check_route_feasible(Route([0, 0]), instance, matrix)


def test_overlong_route_is_infeasible() -> None:
    instance = make_instance(max_distance=5.0)
    matrix = build_distance_matrix(instance)
    assert not check_route_feasible(Route([0, 1, 0]), instance, matrix)


def test_duplicate_visit_is_infeasible() -> None:
    instance = make_instance()
    matrix = build_distance_matrix(instance)
    solution = TOPSolution([Route([0, 1, 0]), Route([0, 1, 0])])
    result = validate_solution(solution, instance, matrix)
    assert not result["feasible"]
    assert any("more than once" in item for item in result["violations"])
