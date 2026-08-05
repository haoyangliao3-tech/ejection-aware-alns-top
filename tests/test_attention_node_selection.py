import random

from top_alns.alns.candidates import (
    AttentionCandidatePoolConfig,
    build_attention_repair_candidates,
)
from top_alns.alns.node_selection import select_next_node
from top_alns.distance import build_distance_matrix
from top_alns.models import Node, Route, TOPInstance, TOPSolution


def _instance() -> TOPInstance:
    return TOPInstance(
        nodes={
            0: Node(0, 0.0, 0.0, 0.0),
            1: Node(1, 9.0, 0.0, 100.0),
            2: Node(2, 1.0, 0.0, 10.0),
            3: Node(3, 2.0, 0.0, 20.0),
        },
        depot_id=0,
        vehicle_count=1,
        max_distance=30.0,
    )


def test_spatial_bucket_prevents_lower_reward_node_starvation() -> None:
    instance = _instance()
    matrix = build_distance_matrix(instance)
    partial = TOPSolution([Route([0, 0])])
    candidates = build_attention_repair_candidates(
        partial,
        [],
        instance,
        matrix,
        [0.0],
        random.Random(0),
        AttentionCandidatePoolConfig(
            bucket_size=1,
            include_reward=False,
            include_reward_density=False,
            include_spatial=True,
            include_regret=False,
            include_random=False,
        ),
    )
    assert candidates == {2}


def test_node_selection_does_not_choose_route_or_position() -> None:
    instance = _instance()
    matrix = build_distance_matrix(instance)
    partial = TOPSolution([Route([0, 0])])
    selected = select_next_node(
        "dynamic_profit_time",
        {1, 2},
        [],
        partial,
        [0.0],
        instance,
        matrix,
        random.Random(0),
    )
    assert selected == 1
