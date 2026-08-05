from pathlib import Path

from top_alns.distance import build_distance_matrix
from top_alns.parser import parse_instance

SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    instance = parse_instance(SAMPLE)
    matrix = build_distance_matrix(instance)
    for first in instance.nodes:
        assert matrix[first][first] == 0.0
        for second in instance.nodes:
            assert matrix[first][second] == matrix[second][first]
