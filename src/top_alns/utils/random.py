"""Central random-number generator construction."""

from __future__ import annotations

import random


def create_rng(seed: int | None) -> random.Random:
    return random.Random(seed)
