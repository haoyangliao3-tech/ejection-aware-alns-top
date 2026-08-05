"""Parser for the project's compact TOP instance format."""

from __future__ import annotations

from pathlib import Path

from .models import Node, TOPInstance


def parse_instance(path: str | Path) -> TOPInstance:
    """Read and validate a TOP instance from *path*."""
    instance_path = Path(path)
    try:
        raw_lines = instance_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read instance '{instance_path}': {exc}") from exc

    lines = [
        (number, line.strip())
        for number, line in enumerate(raw_lines, start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("Instance is empty")

    first_fields = lines[0][1].replace(";", " ").split()
    if first_fields and first_fields[0].lower() == "n":
        from .benchmarks.parser import parse_published_instance

        return parse_published_instance(instance_path)

    header_number, header = lines[0]
    parts = header.split()
    if len(parts) != 3:
        raise ValueError(
            f"Line {header_number}: expected 'VEHICLES max_distance depot_id'"
        )
    try:
        vehicle_count = int(parts[0])
        max_distance = float(parts[1])
        depot_id = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Line {header_number}: invalid header value") from exc
    if vehicle_count <= 0:
        raise ValueError("vehicle_count must be positive")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative")

    nodes: dict[int, Node] = {}
    for line_number, line in lines[1:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"Line {line_number}: expected 'node_id x y reward'"
            )
        try:
            node_id = int(fields[0])
            x, y, reward = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid node value") from exc
        if node_id in nodes:
            raise ValueError(f"Line {line_number}: duplicate node_id {node_id}")
        if reward < 0:
            raise ValueError(
                f"Line {line_number}: reward must be non-negative for node {node_id}"
            )
        nodes[node_id] = Node(node_id, x, y, reward)

    if depot_id not in nodes:
        raise ValueError(f"depot_id {depot_id} does not exist in the node list")
    return TOPInstance(nodes, depot_id, vehicle_count, max_distance)
