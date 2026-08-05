"""Metrics for repeated solver runs."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean, pstdev
from typing import Any


def gap_to_bks(score: float, bks: float) -> float:
    """Return the percentage maximization gap to the best-known score."""
    if bks == 0:
        if score == 0:
            return 0.0
        raise ValueError("bks must be non-zero when score is non-zero")
    return 100.0 * (bks - score) / bks


def summarize_runs(results: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not results:
        return {"run_count": 0}
    rewards = [float(result["best_reward"]) for result in results]
    runtimes = [float(result["runtime_seconds"]) for result in results]
    return {
        "run_count": len(results),
        "best_reward": max(rewards),
        "mean_reward": mean(rewards),
        "reward_std": pstdev(rewards),
        "mean_runtime_seconds": mean(runtimes),
    }
