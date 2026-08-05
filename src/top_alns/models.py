"""Core data structures used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Node:
    node_id: int
    x: float
    y: float
    reward: float


@dataclass(slots=True)
class TOPInstance:
    nodes: dict[int, Node]
    depot_id: int
    vehicle_count: int
    max_distance: float
    end_depot_id: int | None = None

    @property
    def route_end_id(self) -> int:
        return (
            self.depot_id
            if self.end_depot_id is None
            else self.end_depot_id
        )

    @property
    def depot_ids(self) -> set[int]:
        return {self.depot_id, self.route_end_id}


@dataclass(slots=True)
class Route:
    node_ids: list[int]

    def copy(self) -> Route:
        return Route(self.node_ids.copy())


@dataclass(slots=True)
class TOPSolution:
    routes: list[Route]
    total_reward: float = 0.0
    total_distance: float = 0.0
    visited_nodes: set[int] = field(default_factory=set)

    def copy(self) -> TOPSolution:
        return TOPSolution(
            routes=[route.copy() for route in self.routes],
            total_reward=self.total_reward,
            total_distance=self.total_distance,
            visited_nodes=self.visited_nodes.copy(),
        )
