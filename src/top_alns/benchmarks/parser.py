"""Parser for the Chao (1996) and Dang (2013) TOP instance format."""

from __future__ import annotations

from pathlib import Path
import re

from ..models import Node, TOPInstance


def _header_value(line: str, expected_name: str) -> str:
    fields = [field for field in re.split(r"[;\s]+", line.strip()) if field]
    if len(fields) != 2 or fields[0].lower() != expected_name:
        raise ValueError(
            f"expected benchmark header '{expected_name} <value>', got '{line}'"
        )
    return fields[1]


def parse_published_instance(path: str | Path) -> TOPInstance:
    """Parse one published TOP instance with distinct start/end depots."""
    instance_path = Path(path)
    try:
        lines = [
            line.strip()
            for line in instance_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise ValueError(
            f"Cannot read benchmark instance '{instance_path}': {exc}"
        ) from exc
    if len(lines) < 4:
        raise ValueError("benchmark instance is incomplete")

    try:
        node_count = int(_header_value(lines[0], "n"))
        vehicle_count = int(_header_value(lines[1], "m"))
        max_distance = float(_header_value(lines[2], "tmax"))
    except ValueError as exc:
        raise ValueError(
            f"invalid benchmark header in '{instance_path}'"
        ) from exc
    if node_count < 2:
        raise ValueError("benchmark must contain start and end depots")
    if vehicle_count <= 0:
        raise ValueError("vehicle_count must be positive")
    if max_distance < 0:
        raise ValueError("tmax must be non-negative")

    node_lines = lines[3:]
    if len(node_lines) != node_count:
        raise ValueError(
            f"declared n={node_count}, but found {len(node_lines)} nodes"
        )
    nodes: dict[int, Node] = {}
    for node_id, line in enumerate(node_lines):
        fields = [
            field for field in re.split(r"[;\s]+", line) if field
        ]
        if len(fields) != 3:
            raise ValueError(
                f"node row {node_id} must contain X Y P, got '{line}'"
            )
        try:
            x, y, reward = map(float, fields)
        except ValueError as exc:
            raise ValueError(
                f"invalid numeric value in node row {node_id}"
            ) from exc
        if reward < 0:
            raise ValueError(
                f"reward must be non-negative for node {node_id}"
            )
        nodes[node_id] = Node(node_id, x, y, reward)

    return TOPInstance(
        nodes=nodes,
        depot_id=0,
        vehicle_count=vehicle_count,
        max_distance=max_distance,
        end_depot_id=node_count - 1,
    )
