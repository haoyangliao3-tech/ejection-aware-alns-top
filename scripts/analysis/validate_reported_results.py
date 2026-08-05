"""Recompute the manuscript aggregates and paired comparator statistics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


EXPECTED_ROWS = {
    "benchmark_bks": 82,
    "fixed_iteration": 1640,
    "symmetric_wall_clock": 656,
    "runtime_budget": 1968,
    "sensitivity": 126,
    "focused_mechanism": 126,
    "qa_summary": 13,
    "reported_statistics": 16,
}


def gap(row: dict[str, object]) -> float:
    return float(row["BKS_Gap_Percent_Source"])


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_algorithm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_algorithm[str(row["Algorithm"])].append(row)
    result: dict[str, dict[str, float]] = {}
    for algorithm, selected in by_algorithm.items():
        by_instance: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            by_instance[str(row["Instance"])].append(row)
        if len(by_instance) != 82 or any(len(items) != 4 for items in by_instance.values()):
            raise AssertionError(f"{algorithm}: expected 82 instances x 4 seeds")
        best_gaps = [min(gap(row) for row in items) for items in by_instance.values()]
        mean_gaps = [
            sum(gap(row) for row in items) / len(items)
            for items in by_instance.values()
        ]
        runtimes = [float(row["Runtime_Seconds"]) for row in selected]
        completed = [float(row["Completed_Iterations"]) for row in selected]
        result[algorithm] = {
            "Instances": 82.0,
            "Mean_Best_Gap_Percent": float(np.mean(best_gaps)),
            "Mean_Seed_Gap_Percent": float(np.mean(mean_gaps)),
            "BKS_Hits": float(sum(value <= 1e-12 for value in best_gaps)),
            "Mean_Runtime_Seconds_Per_Seed": float(np.mean(runtimes)),
            "Mean_Completed_Iterations_Per_Seed": float(np.mean(completed)),
        }
    return result


def instance_gaps(
    rows: list[dict[str, object]], algorithm: str
) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["Algorithm"] == algorithm:
            grouped[str(row["Instance"])].append(gap(row))
    if len(grouped) != 82 or any(len(values) != 4 for values in grouped.values()):
        raise AssertionError(f"Incomplete paired data for {algorithm}")
    return {
        instance: (min(values), float(np.mean(values)))
        for instance, values in grouped.items()
    }


def paired_statistics(
    rows: list[dict[str, object]], reference: str, comparators: list[str]
) -> list[dict[str, float | str]]:
    maps = {
        algorithm: instance_gaps(rows, algorithm)
        for algorithm in [reference, *comparators]
    }
    output: list[dict[str, float | str]] = []
    for outcome_index, outcome in enumerate(("Best seed", "Mean seed")):
        working: list[dict[str, float | str]] = []
        raw_p_values: list[float] = []
        for comparator in comparators:
            differences = np.asarray(
                [
                    maps[comparator][instance][outcome_index]
                    - maps[reference][instance][outcome_index]
                    for instance in maps[reference]
                ]
            )
            nonzero = differences[np.abs(differences) > 1e-12]
            ranks = rankdata(np.abs(nonzero), method="average")
            rank_biserial = float(
                (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum())
                / ranks.sum()
            )
            test = wilcoxon(
                differences,
                zero_method="wilcox",
                alternative="two-sided",
            )
            raw_p_values.append(float(test.pvalue))
            working.append(
                {
                    "Comparator": comparator,
                    "Outcome": outcome,
                    "W": float(test.statistic),
                    "rank_biserial": rank_biserial,
                }
            )
        for row, adjusted in zip(working, holm_adjust(raw_p_values), strict=True):
            row["p_Holm"] = adjusted
            output.append(row)
    return output


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/results/normalized_results.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    for name, expected in EXPECTED_ROWS.items():
        actual = len(payload[name])
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, found {actual}")

    for table in (
        "fixed_iteration",
        "symmetric_wall_clock",
        "runtime_budget",
        "sensitivity",
        "focused_mechanism",
    ):
        for row in payload[table]:
            expected_gap = 100.0 * (float(row["BKS"]) - float(row["Reward"])) / float(row["BKS"])
            assert_close(gap(row), expected_gap, 1e-10, f"{table} gap")
            if not bool(row["Feasible"]):
                raise AssertionError(f"Infeasible reported run in {table}")

    reported_qa = {
        (row["Experiment"], row["Algorithm"]): row
        for row in payload["qa_summary"]
    }
    for experiment, table in (
        ("Fixed iteration", "fixed_iteration"),
        ("Symmetric wall clock", "symmetric_wall_clock"),
        ("Runtime budget", "runtime_budget"),
    ):
        for algorithm, calculated in aggregate(payload[table]).items():
            reported = reported_qa[(experiment, algorithm)]
            for field, actual in calculated.items():
                assert_close(actual, float(reported[field]), 1e-9, f"{experiment}/{algorithm}/{field}")

    computed_stats: list[dict[str, float | str]] = []
    computed_stats.extend(
        {"Experiment": "Fixed iteration", **row}
        for row in paired_statistics(
            payload["fixed_iteration"], "Ejection ON", ["GRASP", "ILS", "VNS"]
        )
    )
    computed_stats.extend(
        {"Experiment": "Runtime budget", **row}
        for row in paired_statistics(
            payload["runtime_budget"],
            "Ejection ON fixed-iteration reference",
            ["GRASP", "ILS", "VNS", "PyVRP", "Kim augmented LNS"],
        )
    )
    reported_stats = {
        (row["Experiment"], row["Comparator"], row["Outcome"]): row
        for row in payload["reported_statistics"]
    }
    for calculated in computed_stats:
        key = (
            calculated["Experiment"],
            calculated["Comparator"],
            calculated["Outcome"],
        )
        reported = reported_stats[key]
        assert_close(float(calculated["W"]), float(reported["W"]), 1e-9, f"{key}/W")
        assert_close(
            float(calculated["p_Holm"]), float(reported["p_Holm"]), 5e-10, f"{key}/p_Holm"
        )
        assert_close(
            float(calculated["rank_biserial"]),
            float(reported["rank_biserial"]),
            5e-4,
            f"{key}/rank_biserial",
        )

    print("PASS: row counts, gap calculations, feasibility, aggregates, and paired statistics")


if __name__ == "__main__":
    main()
