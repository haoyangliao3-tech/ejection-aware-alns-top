from pathlib import Path

import pytest

from top_alns.parser import parse_instance

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_sample_instance_is_parsed() -> None:
    instance = parse_instance(SAMPLE)
    assert instance.vehicle_count == 2
    assert instance.max_distance == 100.0
    assert instance.depot_id == 0
    assert len(instance.nodes) == 4


def test_duplicate_node_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.txt"
    path.write_text("1 10 0\n0 0 0 0\n0 1 1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        parse_instance(path)
