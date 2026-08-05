"""Acceptance criteria shared by ALNS variants."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
import random


def geometric_cooling_rate(
    initial_temperature: float,
    minimum_temperature: float,
    iterations: int,
) -> float:
    """Return a rate that reaches ``minimum_temperature`` at the run end."""
    if initial_temperature <= 0 or minimum_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if minimum_temperature > initial_temperature:
        raise ValueError(
            "minimum_temperature cannot exceed initial_temperature"
        )
    if iterations <= 0:
        return 1.0
    return (minimum_temperature / initial_temperature) ** (1.0 / iterations)


@dataclass(frozen=True, slots=True)
class SimulatedAnnealingAcceptance:
    """Reward-based simulated annealing for a maximization problem."""

    initial_temperature: float = 100.0
    cooling_rate: float = 0.995
    minimum_temperature: float = 0.01

    def __post_init__(self) -> None:
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if not 0.0 < self.cooling_rate <= 1.0:
            raise ValueError("cooling_rate must be in (0, 1]")
        if self.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")
        if self.minimum_temperature > self.initial_temperature:
            raise ValueError(
                "minimum_temperature cannot exceed initial_temperature"
            )

    def temperature(self, iteration: int) -> float:
        if iteration < 0:
            raise ValueError("iteration must be non-negative")
        return max(
            self.minimum_temperature,
            self.initial_temperature * self.cooling_rate**iteration,
        )

    def acceptance_probability(
        self,
        current_reward: float,
        candidate_reward: float,
        iteration: int,
    ) -> float:
        reward_delta = candidate_reward - current_reward
        if reward_delta >= 0:
            return 1.0
        return exp(reward_delta / self.temperature(iteration))

    def accept(
        self,
        current_reward: float,
        candidate_reward: float,
        iteration: int,
        rng: random.Random,
        current_distance: float | None = None,
        candidate_distance: float | None = None,
    ) -> bool:
        if (
            abs(candidate_reward - current_reward) <= 1e-9
            and current_distance is not None
            and candidate_distance is not None
        ):
            return candidate_distance <= current_distance + 1e-9
        probability = self.acceptance_probability(
            current_reward, candidate_reward, iteration
        )
        return probability >= 1.0 or rng.random() < probability
