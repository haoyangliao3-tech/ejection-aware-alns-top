"""Validate the corrected Dang+Chao workers=2 public result bundle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_ROWS = {
    "benchmark_bks": 239,
    "fixed_iteration": 4780,
    "symmetric_wall_clock": 1912,
    "runtime_budget_baselines": 3824,
    "runtime_budgets": 239,
    "sensitivity": 252,
    "focused_mechanism": 252,
    "qa_summary": 22,
    "reported_statistics": 36,
    "component_statistics": 12,
}
EXPECTED_INSTANCES = {"Dang": 82, "Chao": 157}


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected}")


def validate_run_table(rows: list[dict], name: str, expected_algorithms: set[str]) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        expected_gap = 100.0 * (float(row["BKS"]) - float(row["Reward"])) / float(row["BKS"])
        assert_close(float(row["BKS_Gap_Percent"]), expected_gap, 1e-10, f"{name} gap")
        if not bool(row["Feasible"]):
            raise AssertionError(f"Infeasible run in {name}: {row}")
        if int(row["Batch_Workers"]) != 2:
            raise AssertionError(f"Non-unified worker setting in {name}: {row}")
        grouped[(str(row["Benchmark"]), str(row["Algorithm"]), str(row["Instance"]))].append(row)
    for benchmark, count in EXPECTED_INSTANCES.items():
        algorithms = {key[1] for key in grouped if key[0] == benchmark}
        if algorithms != expected_algorithms:
            raise AssertionError(f"{name}/{benchmark}: algorithms {algorithms}")
        instances = {key[2] for key in grouped if key[0] == benchmark}
        if len(instances) != count:
            raise AssertionError(f"{name}/{benchmark}: expected {count} instances, found {len(instances)}")
        for algorithm in algorithms:
            selected = [items for key, items in grouped.items() if key[0] == benchmark and key[1] == algorithm]
            if len(selected) != count or any(len(items) != 4 for items in selected):
                raise AssertionError(f"{name}/{benchmark}/{algorithm}: expected {count} x 4 seeds")


def recompute_qa(rows: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["Benchmark"], row["Algorithm"], row["Instance"])].append(row)
    by_algorithm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (benchmark, algorithm, _), items in grouped.items():
        bks = float(items[0]["BKS"])
        rewards = [float(item["Reward"]) for item in items]
        by_algorithm[(benchmark, algorithm)].append({
            "best_gap": 100.0 * (bks - max(rewards)) / bks,
            "mean_gap": 100.0 * (bks - float(np.mean(rewards))) / bks,
            "runtime": float(np.mean([float(item["Runtime_Seconds"]) for item in items])),
            "iterations": float(np.mean([float(item["Completed_Iterations"]) for item in items])),
        })
    result = {}
    for key, items in by_algorithm.items():
        result[key] = {
            "Best_Seed_Gap_Percent": float(np.mean([item["best_gap"] for item in items])),
            "Mean_Seed_Gap_Percent": float(np.mean([item["mean_gap"] for item in items])),
            "BKS_Hits": float(sum(item["best_gap"] <= 1e-12 for item in items)),
            "Wall_Clock_Runtime_Per_Seed": float(np.mean([item["runtime"] for item in items])),
            "Mean_Completed_Iterations": float(np.mean([item["iterations"] for item in items])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/results/normalized_results.json"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for table, expected in EXPECTED_ROWS.items():
        if len(payload[table]) != expected:
            raise AssertionError(f"{table}: expected {expected}, found {len(payload[table])}")

    validate_run_table(
        payload["fixed_iteration"], "fixed_iteration",
        {"Ejection ON", "Ejection OFF", "GRASP", "ILS", "VNS"},
    )
    validate_run_table(
        payload["symmetric_wall_clock"], "symmetric_wall_clock",
        {"Ejection ON", "Ejection OFF"},
    )
    validate_run_table(
        payload["runtime_budget_baselines"], "runtime_budget_baselines",
        {"GRASP", "ILS", "VNS", "PyVRP"},
    )
    if any(not bool(row["Within_Budget"]) for row in payload["symmetric_wall_clock"]):
        raise AssertionError("An Ejection ON/OFF run exceeded its recorded budget")
    if any(not bool(row["Within_Budget"]) for row in payload["runtime_budget_baselines"]):
        raise AssertionError("A comparator run exceeded its recorded budget")

    reported = {
        (row["Protocol"], row["Benchmark"], row["Algorithm"]): row
        for row in payload["qa_summary"]
    }
    for protocol, rows in (
        ("Fixed iteration", payload["fixed_iteration"]),
        ("Equal time", payload["symmetric_wall_clock"] + payload["runtime_budget_baselines"]),
    ):
        for (benchmark, algorithm), calculated in recompute_qa(rows).items():
            row = reported[(protocol, benchmark, algorithm)]
            for field, actual in calculated.items():
                assert_close(actual, float(row[field]), 1e-9, f"{protocol}/{benchmark}/{algorithm}/{field}")

    expected_stats = {
        ("Fixed iteration", "Dang"): 8,
        ("Fixed iteration", "Chao"): 8,
        ("Equal time", "Dang"): 10,
        ("Equal time", "Chao"): 10,
    }
    for key, count in expected_stats.items():
        found = sum(row["Protocol"] == key[0] and row["Benchmark"] == key[1] for row in payload["reported_statistics"])
        if found != count:
            raise AssertionError(f"reported_statistics/{key}: expected {count}, found {found}")

    print("PASS: corrected row counts, 4-seed coverage, workers=2, gaps, feasibility, budgets, and aggregates")


if __name__ == "__main__":
    main()
