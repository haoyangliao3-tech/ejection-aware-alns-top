"""TOP adapter for the MIT-licensed PyVRP 0.13.4 package."""

from __future__ import annotations

import math
from time import perf_counter

from pyvrp import Model
from pyvrp.stop import MaxRuntime

from ..distance import build_distance_matrix
from ..feasibility import check_solution_feasible
from ..models import Route, TOPInstance, TOPSolution
from ..solution import update_solution_metrics


DISTANCE_SCALE = 1_000
PRIZE_SCALE = 1_000


def solve_pyvrp_top(
    instance: TOPInstance,
    *,
    seed: int,
    time_limit_seconds: float,
) -> tuple[TOPSolution, int]:
    """Solve TOP with PyVRP, including model construction in the time limit."""
    if time_limit_seconds < 0.0:
        raise ValueError("time_limit_seconds must be non-negative")
    started = perf_counter()
    model = Model()
    start = instance.nodes[instance.depot_id]
    start_depot = model.add_depot(start.x, start.y, name="start depot")
    if instance.route_end_id == instance.depot_id:
        end_depot = start_depot
        locations = [start_depot]
        mapped_ids = [instance.depot_id]
    else:
        end = instance.nodes[instance.route_end_id]
        end_depot = model.add_depot(end.x, end.y, name="end depot")
        locations = [start_depot, end_depot]
        mapped_ids = [instance.depot_id, instance.route_end_id]

    customer_ids = sorted(set(instance.nodes) - instance.depot_ids)
    visit_to_node_id: dict[int, int] = {}
    for node_id in customer_ids:
        node = instance.nodes[node_id]
        client = model.add_client(
            node.x,
            node.y,
            prize=int(round(node.reward * PRIZE_SCALE)),
            required=False,
            name=str(node_id),
        )
        visit_to_node_id[len(locations)] = node_id
        locations.append(client)
        mapped_ids.append(node_id)

    model.add_vehicle_type(
        num_available=instance.vehicle_count,
        start_depot=start_depot,
        end_depot=end_depot,
        max_distance=int(math.floor(instance.max_distance * DISTANCE_SCALE)),
        unit_distance_cost=0,
        name="TOP vehicles",
    )
    matrix = build_distance_matrix(instance)
    for from_index, from_location in enumerate(locations):
        from_id = mapped_ids[from_index]
        for to_index, to_location in enumerate(locations):
            to_id = mapped_ids[to_index]
            distance = int(
                math.ceil(matrix[from_id][to_id] * DISTANCE_SCALE - 1e-12)
            )
            model.add_edge(from_location, to_location, distance=distance)

    remaining = time_limit_seconds - (perf_counter() - started)
    external_routes: list[list[int]] = []
    completed_iterations = 0
    if remaining > 0.0:
        result = model.solve(
            MaxRuntime(remaining),
            seed=seed,
            collect_stats=False,
            display=False,
        )
        completed_iterations = int(result.num_iterations)
        external_routes = [list(route.visits()) for route in result.best.routes()]

    routes: list[Route] = []
    seen: set[int] = set()
    for external_route in external_routes[: instance.vehicle_count]:
        nodes: list[int] = []
        for location_index in external_route:
            try:
                node_id = visit_to_node_id[location_index]
            except KeyError as exc:
                raise RuntimeError(
                    f"PyVRP returned non-client location index {location_index}"
                ) from exc
            if node_id not in seen:
                seen.add(node_id)
                nodes.append(node_id)
        routes.append(Route([instance.depot_id, *nodes, instance.route_end_id]))
    while len(routes) < instance.vehicle_count:
        routes.append(Route([instance.depot_id, instance.route_end_id]))

    solution = update_solution_metrics(TOPSolution(routes=routes), instance, matrix)
    if not check_solution_feasible(solution, instance, matrix):
        raise RuntimeError("PyVRP produced an infeasible TOP solution")
    return solution, completed_iterations
