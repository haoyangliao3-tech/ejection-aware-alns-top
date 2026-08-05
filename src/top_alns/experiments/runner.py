"""Command-line and programmatic runner for Ejection-Aware ALNS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from ..alns.solver import ALNSolver
from ..distance import build_distance_matrix
from ..feasibility import check_solution_feasible
from ..parser import parse_instance


def run_single_instance(
    instance_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Solve one TOP instance and optionally write a JSON result."""
    settings = dict(config or {})
    output_path = settings.pop("output_path", None)
    solver = ALNSolver(**settings)
    instance = parse_instance(instance_path)

    started = perf_counter()
    solution = solver.solve(instance)
    runtime_seconds = perf_counter() - started
    distance_matrix = build_distance_matrix(instance)

    result: dict[str, Any] = {
        "instance": Path(instance_path).name,
        "configuration": settings,
        "total_reward": solution.total_reward,
        "total_distance": solution.total_distance,
        "routes": [route.node_ids for route in solution.routes],
        "visited_nodes": sorted(solution.visited_nodes),
        "feasible": check_solution_feasible(
            solution,
            instance,
            distance_matrix,
        ),
        "runtime_seconds": runtime_seconds,
        "search_diagnostics": solver.last_run_stats,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ejection-Aware Adaptive Large Neighborhood Search on one "
            "Team Orienteering Problem instance."
        )
    )
    parser.add_argument("instance", help="Path to a TOP instance file")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--removal-rate", type=float, default=0.2)
    parser.add_argument("--minimum-removal-rate", type=float, default=0.05)
    parser.add_argument("--initial-temperature", type=float, default=20.0)
    parser.add_argument("--minimum-temperature", type=float, default=0.01)
    parser.add_argument("--cooling-rate", type=float)
    parser.add_argument("--ejection-attempts", type=int, default=100)
    parser.add_argument("--ejection-positions", type=int, default=3)
    parser.add_argument("--ejection-customers", type=int, default=2)
    parser.add_argument("--two-opt-passes", type=int, default=1)
    parser.add_argument(
        "--ejection-ranking",
        choices=("removal_density", "random"),
        default="removal_density",
    )
    parser.add_argument(
        "--ejection-off",
        action="store_true",
        help="Disable the complete Ejection-Aware Repair module.",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Collect repair-level ejection telemetry.",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = {
        "max_iterations": args.iterations,
        "random_seed": args.seed,
        "removal_rate": args.removal_rate,
        "minimum_removal_rate": args.minimum_removal_rate,
        "initial_temperature": args.initial_temperature,
        "minimum_temperature": args.minimum_temperature,
        "cooling_rate": args.cooling_rate,
        "enable_ejection": not args.ejection_off,
        "ejection_max_attempts": args.ejection_attempts,
        "ejection_max_positions": args.ejection_positions,
        "ejection_max_nodes": args.ejection_customers,
        "ejection_two_opt_passes": args.two_opt_passes,
        "ejection_removal_ranking": args.ejection_ranking,
        "collect_ejection_telemetry": args.telemetry,
        "output_path": args.output,
    }
    result = run_single_instance(args.instance, config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
