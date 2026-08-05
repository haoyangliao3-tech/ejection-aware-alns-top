"""Distance calculations."""

from __future__ import annotations

from math import hypot

from .models import Node, TOPInstance

DistanceMatrix = dict[int, dict[int, float]]


def euclidean_distance(node_a: Node, node_b: Node) -> float:
    return hypot(node_a.x - node_b.x, node_a.y - node_b.y)


def build_distance_matrix(instance: TOPInstance) -> DistanceMatrix:
    matrix: DistanceMatrix = {node_id: {} for node_id in instance.nodes}
    node_ids = list(instance.nodes)
    for index, node_a_id in enumerate(node_ids):
        matrix[node_a_id][node_a_id] = 0.0
        for node_b_id in node_ids[index + 1 :]:
            distance = euclidean_distance(
                instance.nodes[node_a_id], instance.nodes[node_b_id]
            )
            matrix[node_a_id][node_b_id] = distance
            matrix[node_b_id][node_a_id] = distance
    return matrix
