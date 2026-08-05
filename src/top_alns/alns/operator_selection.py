"""Lightweight adaptive roulette-wheel operator selection."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Generic, Mapping, TypeVar

OperatorT = TypeVar("OperatorT")


@dataclass(slots=True)
class OperatorScore:
    weight: float = 1.0
    uses: int = 0
    accumulated_reward: float = 0.0


class AdaptiveOperatorSelector(Generic[OperatorT]):
    def __init__(
        self,
        operators: Mapping[str, OperatorT],
        reaction_factor: float = 0.2,
    ) -> None:
        if not operators:
            raise ValueError("at least one operator is required")
        if not 0.0 < reaction_factor <= 1.0:
            raise ValueError("reaction_factor must be in (0, 1]")
        self.operators = dict(operators)
        self.scores = {name: OperatorScore() for name in operators}
        self.reaction_factor = reaction_factor

    def select(self, rng: random.Random) -> tuple[str, OperatorT]:
        names = list(self.operators)
        weights = [max(self.scores[name].weight, 1e-9) for name in names]
        name = rng.choices(names, weights=weights, k=1)[0]
        return name, self.operators[name]

    def update(self, name: str, reward: float) -> None:
        score = self.scores[name]
        score.uses += 1
        score.accumulated_reward += reward

    def end_segment(self) -> None:
        """Update weights from segment averages, then clear observations."""
        for score in self.scores.values():
            if score.uses:
                target = max(score.accumulated_reward / score.uses, 0.0)
                score.weight = (
                    (1.0 - self.reaction_factor) * score.weight
                    + self.reaction_factor * target
                )
            score.uses = 0
            score.accumulated_reward = 0.0
