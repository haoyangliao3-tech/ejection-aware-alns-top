import random
from pathlib import Path

import pytest

from top_alns.alns.acceptance import (
    SimulatedAnnealingAcceptance,
    geometric_cooling_rate,
)
from top_alns.alns.solver import ALNSolver
from top_alns.parser import parse_instance


SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "sample_top_instance.txt"


def test_better_and_equal_rewards_are_always_accepted() -> None:
    acceptance = SimulatedAnnealingAcceptance()
    rng = random.Random(0)
    assert acceptance.accept(100.0, 101.0, 0, rng)
    assert acceptance.accept(100.0, 100.0, 0, rng)


def test_worse_reward_acceptance_depends_on_temperature() -> None:
    hot = SimulatedAnnealingAcceptance(
        initial_temperature=100.0,
        cooling_rate=1.0,
        minimum_temperature=0.01,
    )
    cold = SimulatedAnnealingAcceptance(
        initial_temperature=0.01,
        cooling_rate=1.0,
        minimum_temperature=0.01,
    )
    assert hot.accept(100.0, 99.0, 0, random.Random(0))
    assert not cold.accept(100.0, 99.0, 0, random.Random(0))


def test_equal_reward_rejects_a_longer_solution_when_distance_is_known() -> None:
    acceptance = SimulatedAnnealingAcceptance()
    assert not acceptance.accept(
        100.0,
        100.0,
        0,
        random.Random(0),
        current_distance=50.0,
        candidate_distance=51.0,
    )


def test_temperature_cools_to_configured_minimum() -> None:
    acceptance = SimulatedAnnealingAcceptance(
        initial_temperature=10.0,
        cooling_rate=0.5,
        minimum_temperature=1.0,
    )
    assert acceptance.temperature(0) == pytest.approx(10.0)
    assert acceptance.temperature(2) == pytest.approx(2.5)
    assert acceptance.temperature(100) == pytest.approx(1.0)


def test_automatic_rate_reaches_minimum_at_iteration_budget() -> None:
    rate = geometric_cooling_rate(20.0, 0.01, 5_000)
    acceptance = SimulatedAnnealingAcceptance(
        initial_temperature=20.0,
        cooling_rate=rate,
        minimum_temperature=0.01,
    )
    assert acceptance.temperature(5_000) == pytest.approx(0.01)


def test_invalid_annealing_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        SimulatedAnnealingAcceptance(initial_temperature=0.0)
    with pytest.raises(ValueError):
        SimulatedAnnealingAcceptance(cooling_rate=1.1)


def test_alns_supports_wall_clock_calibrated_cooling() -> None:
    solver = ALNSolver(
        max_iterations=2,
        random_seed=0,
        sa_wall_clock_horizon_seconds=10.0,
    )
    solution = solver.solve(parse_instance(SAMPLE))

    assert solution.routes
    assert solver.last_run_stats["sa_schedule"] == "wall_clock_geometric"
    assert solver.last_run_stats["sa_wall_clock_horizon_seconds"] == 10.0


def test_invalid_wall_clock_cooling_horizon_is_rejected() -> None:
    with pytest.raises(ValueError):
        ALNSolver(sa_wall_clock_horizon_seconds=0.0)
    with pytest.raises(ValueError):
        ALNSolver(
            sa_wall_clock_horizon_seconds=1.0,
            sa_cyclic_reheat=True,
        )
