"""Interpretable attention-inspired insertion scoring.

Every component is exposed as an independent function so experiments can set
its weight to zero or replace it without changing the repair operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from ..distance import DistanceMatrix
from ..greedy import insertion_cost
from ..models import Route, TOPInstance, TOPSolution
from ..solution import calculate_route_distance

EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class AttentionWeights:
    """Coefficients corresponding to alpha_1, ..., alpha_5."""

    reward_density: float = 1.0
    # Feasibility is enforced before scoring, so this term is constant.
    insertion_feasibility: float = 0.0
    route_remaining_capacity: float = 1.0
    spatial_compatibility: float = 1.0
    # Route balance is not part of the TOP objective; keep it opt-in.
    route_balance_score: float = 0.0


@dataclass(frozen=True, slots=True)
class AttentionScoreComponents:
    reward_density: float
    insertion_feasibility: float
    route_remaining_capacity: float
    spatial_compatibility: float
    route_balance_score: float

    def weighted_sum(self, weights: AttentionWeights) -> float:
        return (
            weights.reward_density * self.reward_density
            + weights.insertion_feasibility * self.insertion_feasibility
            + weights.route_remaining_capacity * self.route_remaining_capacity
            + weights.spatial_compatibility * self.spatial_compatibility
            + weights.route_balance_score * self.route_balance_score
        )


def normalize_attention_components(
    components: list[AttentionScoreComponents],
) -> list[AttentionScoreComponents]:
    """Min-max normalize one repair decision batch component by component.

    Raw components remain available for diagnostics. Normalization only makes
    their weighted contributions comparable when selecting an insertion.
    """
    if not components:
        return []
    field_names = (
        "reward_density",
        "insertion_feasibility",
        "route_remaining_capacity",
        "spatial_compatibility",
        "route_balance_score",
    )
    bounds = {
        name: (
            min(getattr(component, name) for component in components),
            max(getattr(component, name) for component in components),
        )
        for name in field_names
    }

    def normalized(value: float, name: str) -> float:
        minimum, maximum = bounds[name]
        span = maximum - minimum
        if span <= EPSILON:
            return 1.0
        return (value - minimum) / span

    return [
        AttentionScoreComponents(
            reward_density=normalized(
                component.reward_density, "reward_density"
            ),
            insertion_feasibility=normalized(
                component.insertion_feasibility,
                "insertion_feasibility",
            ),
            route_remaining_capacity=normalized(
                component.route_remaining_capacity,
                "route_remaining_capacity",
            ),
            spatial_compatibility=normalized(
                component.spatial_compatibility,
                "spatial_compatibility",
            ),
            route_balance_score=normalized(
                component.route_balance_score,
                "route_balance_score",
            ),
        )
        for component in components
    ]


@dataclass(slots=True)
class AttentionRouteCache:
    """Incremental route state used by the attention repair operator."""

    route_distances: list[float]
    other_min_utilizations: list[float | None]
    other_max_utilizations: list[float | None]
    other_max_distances: list[float]

    @classmethod
    def from_solution(
        cls,
        solution: TOPSolution,
        instance: TOPInstance,
        distance_matrix: DistanceMatrix,
    ) -> AttentionRouteCache:
        cache = cls(
            route_distances=[
                calculate_route_distance(route, instance, distance_matrix)
                for route in solution.routes
            ],
            other_min_utilizations=[],
            other_max_utilizations=[],
            other_max_distances=[],
        )
        cache._refresh_balance_extrema(instance)
        return cache

    def _refresh_balance_extrema(self, instance: TOPInstance) -> None:
        if instance.max_distance <= EPSILON:
            utilizations = [0.0 for _ in self.route_distances]
        else:
            utilizations = [
                min(1.0, max(0.0, distance / instance.max_distance))
                for distance in self.route_distances
            ]
        self.other_min_utilizations = []
        self.other_max_utilizations = []
        self.other_max_distances = []
        for route_index in range(len(self.route_distances)):
            other_utilizations = (
                utilizations[:route_index] + utilizations[route_index + 1 :]
            )
            other_distances = (
                self.route_distances[:route_index]
                + self.route_distances[route_index + 1 :]
            )
            self.other_min_utilizations.append(
                min(other_utilizations) if other_utilizations else None
            )
            self.other_max_utilizations.append(
                max(other_utilizations) if other_utilizations else None
            )
            self.other_max_distances.append(
                max(other_distances, default=0.0)
            )

    def apply_insertion(
        self,
        route_index: int,
        additional_distance: float,
        instance: TOPInstance,
    ) -> None:
        self.route_distances[route_index] += additional_distance
        self._refresh_balance_extrema(instance)

    def set_route_distance(
        self,
        route_index: int,
        new_distance: float,
        instance: TOPInstance,
    ) -> None:
        """Overwrite one route's cached distance after a non-incremental edit.

        Ejection moves rewrite a whole route (remove nodes, then 2-opt), so the
        change cannot be expressed as a single additional-distance increment.
        """
        self.route_distances[route_index] = new_distance
        self._refresh_balance_extrema(instance)


def reward_density_from_increment(
    reward: float, additional_distance: float
) -> float:
    return reward / max(additional_distance, EPSILON)


def insertion_feasibility_from_distance(
    new_route_distance: float, instance: TOPInstance
) -> float:
    return float(new_route_distance <= instance.max_distance + 1e-9)


def route_remaining_capacity_from_distance(
    new_route_distance: float, instance: TOPInstance
) -> float:
    remaining = instance.max_distance - new_route_distance
    if instance.max_distance <= EPSILON:
        return float(remaining >= -1e-9)
    return min(1.0, max(0.0, remaining / instance.max_distance))


def spatial_compatibility_from_neighbor_distance(
    mean_neighbor_distance: float, instance: TOPInstance
) -> float:
    if instance.max_distance <= EPSILON:
        return float(mean_neighbor_distance <= EPSILON)
    return 1.0 / (
        1.0 + mean_neighbor_distance / instance.max_distance
    )


def route_balance_score_from_cache(
    new_route_distance: float,
    route_index: int,
    cache: AttentionRouteCache,
    instance: TOPInstance,
) -> float:
    if instance.max_distance <= EPSILON:
        projected_max = max(
            new_route_distance,
            cache.other_max_distances[route_index],
        )
        return float(projected_max <= EPSILON)

    new_utilization = min(
        1.0, max(0.0, new_route_distance / instance.max_distance)
    )
    other_min = cache.other_min_utilizations[route_index]
    other_max = cache.other_max_utilizations[route_index]
    if other_min is None or other_max is None:
        return 1.0
    projected_min = min(new_utilization, other_min)
    projected_max = max(new_utilization, other_max)
    return max(0.0, 1.0 - (projected_max - projected_min))


def incremental_attention_insertion_score(
    *,
    reward: float,
    additional_distance: float,
    new_route_distance: float,
    mean_neighbor_distance: float,
    route_index: int,
    cache: AttentionRouteCache,
    instance: TOPInstance,
    weights: AttentionWeights,
) -> float:
    """Evaluate one candidate using only cached and incremental values."""
    feasibility = insertion_feasibility_from_distance(
        new_route_distance, instance
    )
    if feasibility == 0.0:
        return -inf
    return (
        weights.reward_density
        * reward_density_from_increment(reward, additional_distance)
        + weights.insertion_feasibility * feasibility
        + weights.route_remaining_capacity
        * route_remaining_capacity_from_distance(
            new_route_distance, instance
        )
        + weights.spatial_compatibility
        * spatial_compatibility_from_neighbor_distance(
            mean_neighbor_distance, instance
        )
        + weights.route_balance_score
        * route_balance_score_from_cache(
            new_route_distance, route_index, cache, instance
        )
    )


def incremental_attention_score_components(
    *,
    reward: float,
    additional_distance: float,
    new_route_distance: float,
    mean_neighbor_distance: float,
    route_index: int,
    cache: AttentionRouteCache,
    instance: TOPInstance,
) -> AttentionScoreComponents:
    """Return raw components using cached route state and O(1) increments."""
    return AttentionScoreComponents(
        reward_density=reward_density_from_increment(
            reward, additional_distance
        ),
        insertion_feasibility=insertion_feasibility_from_distance(
            new_route_distance, instance
        ),
        route_remaining_capacity=route_remaining_capacity_from_distance(
            new_route_distance, instance
        ),
        spatial_compatibility=spatial_compatibility_from_neighbor_distance(
            mean_neighbor_distance, instance
        ),
        route_balance_score=route_balance_score_from_cache(
            new_route_distance, route_index, cache, instance
        ),
    )


def reward_density(
    node_id: int,
    route: Route,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float:
    """Return reward divided by the additional insertion distance."""
    additional = insertion_cost(route, position, node_id, distance_matrix)
    return reward_density_from_increment(
        instance.nodes[node_id].reward, additional
    )


def insertion_feasibility(
    node_id: int,
    route: Route,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float:
    """Return 1 if the insertion respects the route distance limit, else 0."""
    current = calculate_route_distance(route, instance, distance_matrix)
    additional = insertion_cost(route, position, node_id, distance_matrix)
    return insertion_feasibility_from_distance(
        current + additional, instance
    )


def route_remaining_capacity(
    node_id: int,
    route: Route,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float:
    """Return the route's remaining distance ratio after insertion."""
    current = calculate_route_distance(route, instance, distance_matrix)
    additional = insertion_cost(route, position, node_id, distance_matrix)
    return route_remaining_capacity_from_distance(
        current + additional, instance
    )


def spatial_compatibility(
    node_id: int,
    route: Route,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float:
    """Measure normalized proximity to the two nodes around the insertion."""
    previous_id = route.node_ids[position - 1]
    next_id = route.node_ids[position]
    mean_neighbor_distance = (
        distance_matrix[previous_id][node_id]
        + distance_matrix[node_id][next_id]
    ) / 2.0
    return spatial_compatibility_from_neighbor_distance(
        mean_neighbor_distance, instance
    )


def route_balance_score(
    node_id: int,
    solution: TOPSolution,
    route_index: int,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> float:
    """Score balance as one minus the projected route-utilization range."""
    if not solution.routes:
        return 1.0
    projected_distances = [
        calculate_route_distance(route, instance, distance_matrix)
        for route in solution.routes
    ]
    projected_distances[route_index] += insertion_cost(
        solution.routes[route_index], position, node_id, distance_matrix
    )
    if instance.max_distance <= EPSILON:
        return float(max(projected_distances, default=0.0) <= EPSILON)
    utilizations = [
        min(1.0, max(0.0, distance / instance.max_distance))
        for distance in projected_distances
    ]
    return max(0.0, 1.0 - (max(utilizations) - min(utilizations)))


def attention_score_components(
    node_id: int,
    solution: TOPSolution,
    route_index: int,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
) -> AttentionScoreComponents:
    route = solution.routes[route_index]
    return AttentionScoreComponents(
        reward_density=reward_density(
            node_id, route, position, instance, distance_matrix
        ),
        insertion_feasibility=insertion_feasibility(
            node_id, route, position, instance, distance_matrix
        ),
        route_remaining_capacity=route_remaining_capacity(
            node_id, route, position, instance, distance_matrix
        ),
        spatial_compatibility=spatial_compatibility(
            node_id, route, position, instance, distance_matrix
        ),
        route_balance_score=route_balance_score(
            node_id,
            solution,
            route_index,
            position,
            instance,
            distance_matrix,
        ),
    )


def attention_insertion_score(
    node_id: int,
    solution: TOPSolution,
    route_index: int,
    position: int,
    instance: TOPInstance,
    distance_matrix: DistanceMatrix,
    weights: AttentionWeights | None = None,
) -> float:
    """Evaluate the weighted attention-inspired insertion formula."""
    components = attention_score_components(
        node_id,
        solution,
        route_index,
        position,
        instance,
        distance_matrix,
    )
    if components.insertion_feasibility == 0.0:
        return -inf
    return components.weighted_sum(weights or AttentionWeights())
