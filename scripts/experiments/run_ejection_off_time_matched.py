#!/usr/bin/env python
"""Run one Ejection ON/OFF arm under instance-mean reference budgets.

The reference is the existing four-seed, 2,500-iteration Ejection ON batch.
The time stop is enforced by a progress callback at completed-iteration
boundaries, with a small safety reserve to keep measured runtime inside the
reference hard wall.  The optional wall-clock cooling mode maps elapsed time,
rather than completed iterations, from the initial to the minimum temperature.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from top_alns.alns.acceptance import geometric_cooling_rate  # noqa: E402
from top_alns.alns.solver import ALNSolver  # noqa: E402
from top_alns.benchmarks.references import load_published_references  # noqa: E402
from top_alns.benchmarks.runner import discover_instances  # noqa: E402
from top_alns.distance import build_distance_matrix  # noqa: E402
from top_alns.feasibility import validate_solution  # noqa: E402
from top_alns.parser import parse_instance  # noqa: E402


DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "outputs"
    / "ejection_k100_l1_2500_optimized_4seeds_on_off"
    / "ejection_benchmark_20260715T033206Z.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "ejection_off_time_matched_on_reference_82_4seeds_20260727"
)
MAX_SEARCH_ITERATIONS = 2_147_483_647

RAW_FIELDS = (
    "instance",
    "algorithm",
    "ejection",
    "seed",
    "bks",
    "reference_iterations",
    "reference_runtime_seconds",
    "effective_time_limit_seconds",
    "runtime_seconds",
    "runtime_slack_seconds",
    "within_time_budget",
    "timeout_status",
    "timed_out",
    "final_stop_guard_seconds",
    "completed_iterations",
    "reward",
    "bks_gap_percent",
    "distance",
    "visited_node_count",
    "feasible",
    "completed_at_utc",
)


class TimeBudgetReached(RuntimeError):
    """Internal control-flow signal raised after a completed iteration."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row["instance"]).lower(), int(row["seed"]))] = row
    return sorted(latest.values(), key=lambda row: (row["instance"], row["seed"]))


def load_reference(
    path: Path, seeds: list[int], reference_iterations: int
) -> tuple[dict[str, float], dict[tuple[str, int], dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    configuration = report.get("configuration", {})
    if int(configuration.get("iterations", -1)) != reference_iterations:
        raise SystemExit(
            f"Reference is not a {reference_iterations}-iteration batch: {path}"
        )

    selected_seeds = set(seeds)
    on_rows: dict[tuple[str, int], dict[str, Any]] = {}
    by_instance: dict[str, list[float]] = {}
    for row in report.get("raw_runs", []):
        if not bool(row.get("ejection", False)):
            continue
        seed = int(row["seed"])
        if seed not in selected_seeds:
            continue
        instance = str(row["instance"]).lower()
        key = (instance, seed)
        if key in on_rows:
            raise SystemExit(f"Duplicate ON reference row: {instance}/seed{seed}")
        runtime = float(row["runtime_seconds"])
        if runtime <= 0.0:
            raise SystemExit(f"Non-positive ON runtime: {instance}/seed{seed}")
        if not bool(row.get("feasible", False)):
            raise SystemExit(f"Infeasible ON reference row: {instance}/seed{seed}")
        on_rows[key] = row
        by_instance.setdefault(instance, []).append(runtime)

    budgets: dict[str, float] = {}
    for instance, values in by_instance.items():
        if len(values) != len(selected_seeds):
            raise SystemExit(
                f"Expected {len(selected_seeds)} ON seeds for {instance}, found {len(values)}"
            )
        budgets[instance] = sum(values) / len(values)
    return budgets, on_rows


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    instance = parse_instance(job["instance_path"])
    budget = float(job["reference_runtime_seconds"])
    safety = min(
        float(job["maximum_safety_seconds"]),
        max(
            float(job["minimum_safety_seconds"]),
            budget * float(job["safety_fraction"]),
        ),
    )
    effective_limit = budget
    wall_clock_cooling = bool(job["wall_clock_cooling"])
    cooling_rate = (
        None
        if wall_clock_cooling
        else geometric_cooling_rate(
            float(job["initial_temperature"]),
            float(job["minimum_temperature"]),
            int(job["reference_iterations"]),
        )
    )
    solver = ALNSolver(
        max_iterations=MAX_SEARCH_ITERATIONS,
        removal_rate=float(job["removal_rate"]),
        minimum_removal_rate=float(job["minimum_removal_rate"]),
        random_seed=int(job["seed"]),
        initial_temperature=float(job["initial_temperature"]),
        cooling_rate=cooling_rate,
        minimum_temperature=float(job["minimum_temperature"]),
        enable_ejection=bool(job["enable_ejection"]),
        sa_wall_clock_horizon_seconds=(budget if wall_clock_cooling else None),
    )

    state: dict[str, Any] = {
        "best": None,
        "completed_iterations": 0,
        "last_callback_time": None,
        "max_normal_iteration_seconds": 0.0,
        "max_exchange_iteration_seconds": 0.0,
        "final_stop_guard_seconds": safety,
    }
    started = perf_counter()
    deadline = started + effective_limit
    state["last_callback_time"] = started

    def stop_at_iteration_boundary(current: int, total: int) -> None:
        del total
        now = perf_counter()
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        try:
            best = caller.f_locals.get("best") if caller is not None else None
            if best is not None:
                state["best"] = best.copy()
            state["completed_iterations"] = int(current)
            iteration_seconds = now - float(state["last_callback_time"])
            current_used_exchange = bool(
                caller.f_locals.get("should_exchange", False)
                if caller is not None and current > 0
                else False
            )
            if current > 0:
                bucket = (
                    "max_exchange_iteration_seconds"
                    if current_used_exchange
                    else "max_normal_iteration_seconds"
                )
                state[bucket] = max(float(state[bucket]), iteration_seconds)
            iterations_since_best = int(
                caller.f_locals.get("iterations_since_best", 0)
                if caller is not None
                else 0
            )
        finally:
            del caller
            del frame

        state["last_callback_time"] = now
        guard = max(
            safety,
            float(job["normal_guard_multiplier"])
            * float(state["max_normal_iteration_seconds"]),
        )
        next_stagnation = iterations_since_best + 1
        next_may_exchange = (
            next_stagnation >= int(job["exchange_stagnation"])
            and next_stagnation % int(job["exchange_stagnation"]) == 0
        )
        if next_may_exchange:
            guard = max(
                guard,
                float(job["exchange_guard_seconds"]),
                float(job["exchange_guard_multiplier"])
                * float(state["max_exchange_iteration_seconds"]),
            )
        state["final_stop_guard_seconds"] = guard
        if current > 0 and deadline - now <= guard:
            raise TimeBudgetReached

    timed_out = False
    timeout_status = "completed_without_time_stop"
    try:
        solution = solver.solve(
            instance,
            progress_callback=stop_at_iteration_boundary,
        )
    except TimeBudgetReached:
        timed_out = True
        timeout_status = "time_limit_guard_at_iteration_boundary"
        solution = state["best"]
        if solution is None:
            raise RuntimeError("Time stop occurred before a best solution was captured")
    runtime = perf_counter() - started

    matrix = build_distance_matrix(instance)
    validation = validate_solution(solution, instance, matrix)
    bks = float(job["bks"])
    reward = float(solution.total_reward)
    return {
        "instance": job["instance"],
        "algorithm": (
            "ejection_on_time_stopped"
            if bool(job["enable_ejection"])
            else "ejection_off_time_stopped"
        ),
        "ejection": bool(job["enable_ejection"]),
        "seed": int(job["seed"]),
        "bks": bks,
        "reference_iterations": int(job["reference_iterations"]),
        "reference_runtime_seconds": budget,
        "effective_time_limit_seconds": effective_limit,
        "runtime_seconds": runtime,
        "runtime_slack_seconds": budget - runtime,
        "within_time_budget": runtime <= budget,
        "timeout_status": timeout_status,
        "timed_out": timed_out,
        "final_stop_guard_seconds": float(state["final_stop_guard_seconds"]),
        "completed_iterations": int(state["completed_iterations"]),
        "reward": reward,
        "bks_gap_percent": 100.0 * (bks - reward) / bks,
        "distance": float(solution.total_distance),
        "visited_node_count": len(solution.visited_nodes),
        "feasible": bool(validation["feasible"]),
        "completed_at_utc": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", choices=("dang", "chao"), default="dang")
    parser.add_argument("--expected-instance-count", type=int)
    parser.add_argument("--reference-iterations", type=int, default=2500)
    parser.add_argument("--ejection", choices=("on", "off"), default="off")
    parser.add_argument("--wall-clock-cooling", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--reference-budget-seeds", nargs="+", type=int, default=[0, 1, 2, 3]
    )
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--removal-rate", type=float, default=0.20)
    parser.add_argument("--minimum-removal-rate", type=float, default=0.05)
    parser.add_argument("--initial-temperature", type=float, default=20.0)
    parser.add_argument("--minimum-temperature", type=float, default=0.01)
    parser.add_argument("--safety-fraction", type=float, default=0.002)
    parser.add_argument("--minimum-safety-seconds", type=float, default=0.10)
    parser.add_argument("--maximum-safety-seconds", type=float, default=0.50)
    parser.add_argument("--normal-guard-multiplier", type=float, default=1.50)
    parser.add_argument("--exchange-guard-seconds", type=float, default=0.50)
    parser.add_argument("--exchange-guard-multiplier", type=float, default=2.00)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    reference_budget_seeds = list(dict.fromkeys(args.reference_budget_seeds))
    if args.workers <= 0 or args.reference_iterations <= 0:
        raise SystemExit("workers and reference iterations must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("batch size must be positive")

    benchmark_root = args.benchmark_root.resolve()
    instances = discover_instances(benchmark_root, dataset=args.dataset, pattern="*.txt")
    references = load_published_references(benchmark_root)
    dataset_label = "Chao1996" if args.dataset == "chao" else "Dang2013"
    instances = [
        path for path in instances
        if path.stem.lower() in references
        and references[path.stem.lower()].dataset == dataset_label
        and references[path.stem.lower()].best_known is not None
    ]
    if args.instances:
        wanted = {name.lower() for name in args.instances}
        instances = [path for path in instances if path.stem.lower() in wanted]
        missing_names = wanted - {path.stem.lower() for path in instances}
        if missing_names:
            raise SystemExit(f"Instances not found: {sorted(missing_names)}")
    if args.limit is not None:
        instances = instances[: args.limit]
    expected_instance_count = args.expected_instance_count or (157 if args.dataset == "chao" else 82)
    if not args.instances and args.limit is None and len(instances) != expected_instance_count:
        raise SystemExit(
            f"Expected {expected_instance_count} {args.dataset} instances with a "
            f"published BKS, found {len(instances)}"
        )

    reference_path = args.reference_json.resolve()
    budgets, on_rows = load_reference(
        reference_path, reference_budget_seeds, args.reference_iterations
    )
    required = {path.stem.lower() for path in instances}
    missing_budgets = sorted(required - set(budgets))
    if missing_budgets:
        raise SystemExit(f"Missing ON budgets: {missing_budgets[:10]}")

    enable_ejection = args.ejection == "on"
    configuration = {
        "protocol": "single_arm_under_instance_mean_on_2500_reference_budget",
        "comparison_design": "symmetric_time_stopped_arm",
        "benchmark_root": str(benchmark_root),
        "dataset": dataset_label,
        "instance_count": len(instances),
        "instances": [path.stem for path in instances],
        "seeds": seeds,
        "reference_budget_seeds": reference_budget_seeds,
        "reference_iterations": args.reference_iterations,
        "reference_json": str(reference_path),
        "workers": args.workers,
        "removal_rate": args.removal_rate,
        "minimum_removal_rate": args.minimum_removal_rate,
        "initial_temperature": args.initial_temperature,
        "minimum_temperature": args.minimum_temperature,
        "cooling_horizon_iterations": (
            None if args.wall_clock_cooling else args.reference_iterations
        ),
        "cooling_horizon_seconds": (
            "instance_specific_reference_budget"
            if args.wall_clock_cooling
            else None
        ),
        "cooling_schedule": (
            "wall_clock_geometric"
            if args.wall_clock_cooling
            else "iteration_geometric"
        ),
        "enable_ejection": enable_ejection,
        "safety_fraction": args.safety_fraction,
        "minimum_safety_seconds": args.minimum_safety_seconds,
        "maximum_safety_seconds": args.maximum_safety_seconds,
        "normal_guard_multiplier": args.normal_guard_multiplier,
        "exchange_guard_seconds": args.exchange_guard_seconds,
        "exchange_guard_multiplier": args.exchange_guard_multiplier,
        "time_check": "completed_iteration_boundary",
        "algorithm_source_sha256": {
            "alns_solver.py": sha256_file(SRC_ROOT / "top_alns" / "alns" / "solver.py"),
            "alns_repair.py": sha256_file(SRC_ROOT / "top_alns" / "alns" / "repair.py"),
        },
        "batch_size": args.batch_size,
    }
    total = len(instances) * len(seeds)
    preview = {
        **configuration,
        "total_runs": total,
        "mean_instance_budget_seconds": sum(budgets[name] for name in required) / len(required),
        "minimum_instance_budget_seconds": min(budgets[name] for name in required),
        "maximum_instance_budget_seconds": max(budgets[name] for name in required),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    raw_jsonl = output_dir / "raw_runs.jsonl"
    failures_jsonl = output_dir / "failures.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration") != configuration:
            raise SystemExit("Output directory contains a different experiment")

    previous = deduplicate(read_jsonl(raw_jsonl))
    completed = {(str(row["instance"]).lower(), int(row["seed"])) for row in previous}
    jobs: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for path in instances:
        instance_key = path.stem.lower()
        ref = references.get(instance_key)
        if ref is None or not ref.best_known:
            raise SystemExit(f"Missing BKS for {path.stem}")
        budget_rows.append(
            {
                "instance": path.stem,
                "reference_seed_count": len(reference_budget_seeds),
                "reference_runtime_mean_seconds": budgets[instance_key],
                "bks": float(ref.best_known),
            }
        )
        for seed in seeds:
            if (instance_key, seed) in completed:
                continue
            jobs.append(
                {
                    "instance": path.stem,
                    "instance_path": str(path.resolve()),
                    "seed": seed,
                    "bks": float(ref.best_known),
                    "reference_iterations": args.reference_iterations,
                    "reference_runtime_seconds": budgets[instance_key],
                    "removal_rate": args.removal_rate,
                    "minimum_removal_rate": args.minimum_removal_rate,
                    "initial_temperature": args.initial_temperature,
                    "minimum_temperature": args.minimum_temperature,
                    "enable_ejection": enable_ejection,
                    "wall_clock_cooling": args.wall_clock_cooling,
                    "safety_fraction": args.safety_fraction,
                    "minimum_safety_seconds": args.minimum_safety_seconds,
                    "maximum_safety_seconds": args.maximum_safety_seconds,
                    "normal_guard_multiplier": args.normal_guard_multiplier,
                    "exchange_guard_seconds": args.exchange_guard_seconds,
                    "exchange_guard_multiplier": args.exchange_guard_multiplier,
                    "exchange_stagnation": 200,
                    "on_reference_reward": float(on_rows[(instance_key, seed)]["reward"]),
                }
            )

    write_csv(
        output_dir / "budgets.csv",
        budget_rows,
        ("instance", "reference_seed_count", "reference_runtime_mean_seconds", "bks"),
    )
    batch_jobs = jobs[: args.batch_size] if args.batch_size is not None else jobs
    batch_limited = len(batch_jobs) < len(jobs)
    manifest = {
        "configuration": configuration,
        "status": "running",
        "started_or_resumed_at_utc": utc_now(),
        "completed_runs": len(completed),
        "total_runs": total,
    }
    write_json(manifest_path, manifest)

    failures = 0
    done = len(completed)
    with raw_jsonl.open("a", encoding="utf-8", buffering=1) as raw_handle:
        if batch_jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {
                    executor.submit(run_one, job): job for job in batch_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        failures += 1
                        failure = {
                            "instance": job["instance"],
                            "seed": job["seed"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_at_utc": utc_now(),
                        }
                        with failures_jsonl.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        print(
                            f"[FAIL] {job['instance']} seed={job['seed']}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    else:
                        raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        raw_handle.flush()
                        done += 1
                        print(
                            f"[{done:3d}/{total}] {row['instance']:20s} seed={row['seed']} "
                            f"reward={row['reward']:.0f} iter={row['completed_iterations']} "
                            f"t={row['runtime_seconds']:.3f}/{row['reference_runtime_seconds']:.3f}s "
                            f"within={row['within_time_budget']} feasible={row['feasible']}",
                            flush=True,
                        )

    records = deduplicate(read_jsonl(raw_jsonl))
    write_csv(output_dir / "raw_runs.csv", records, RAW_FIELDS)
    write_json(
        output_dir / "results.json",
        {
            "configuration": configuration,
            "budgets": budget_rows,
            "raw_runs": records,
        },
    )
    over_budget = sum(not bool(row["within_time_budget"]) for row in records)
    infeasible = sum(not bool(row["feasible"]) for row in records)
    manifest.update(
        {
            "status": (
                "complete"
                if len(records) == total and failures == 0
                else "incomplete"
            ),
            "finished_at_utc": utc_now(),
            "completed_runs": len(records),
            "over_budget_records": over_budget,
            "infeasible_records": infeasible,
            "failed_runs_this_invocation": failures,
        }
    )
    write_json(manifest_path, manifest)
    print(
        f"Recorded {len(records)}/{total}; over budget={over_budget}; "
        f"infeasible={infeasible}; failures={failures}",
        flush=True,
    )
    if len(records) != total and not (batch_limited and failures == 0):
        raise SystemExit("Experiment incomplete; rerun the same command to resume")


if __name__ == "__main__":
    main()
