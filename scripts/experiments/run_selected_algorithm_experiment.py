"""Run the Kim--Li--Johnson augmented LNS reproduction.

Supported protocols are exactly 2,500 algorithm-specific iterations and
the existing per-instance mean-runtime hard-wall budget.  Runs are appended
immediately and can be resumed with the same command.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median, pstdev
import sys
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from top_alns.benchmarks.references import load_published_references
from top_alns.benchmarks.runner import discover_instances
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import validate_solution
from top_alns.kim_alns import KimALNSConfig, solve_kim_alns
from top_alns.parser import parse_instance


DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
DEFAULT_BUDGET_CSV = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "csv"
    / "runtime_budgets.csv"
)
ALGORITHM_LABELS = {
    "kim_alns": "Kim--Li--Johnson augmented LNS reproduction",
}
EPSILON = 1e-9
RAW_FIELDS = (
    "instance",
    "algorithm",
    "algorithm_label",
    "protocol",
    "seed",
    "requested_iterations",
    "completed_iterations",
    "reference_runtime_seconds",
    "effective_time_limit_seconds",
    "runtime_seconds",
    "runtime_slack_seconds",
    "within_time_budget",
    "timeout",
    "termination_reason",
    "best_reward",
    "bks",
    "bks_gap_percent",
    "total_distance",
    "visited_node_count",
    "feasible",
    "violations",
    "position_updates",
    "swarm_rounds",
    "completed_heuristic_initialisations",
    "pool_size",
    "accepted_pool_updates",
    "completed_at_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataclass_values(value: Any) -> dict[str, Any]:
    return {
        name: getattr(value, name)
        for name in value.__dataclass_fields__
    }


def read_budgets(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    budgets = {
        str(row["instance"]).lower(): float(row["reference_runtime_mean_seconds"])
        for row in rows
    }
    if not budgets or any(value <= 0 for value in budgets.values()):
        raise ValueError("budget file must contain positive instance budgets")
    return budgets


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["instance"]).lower(), int(row["seed"])


def deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {key(row): row for row in rows}
    return sorted(by_key.values(), key=key)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def gap_percent(score: float, bks: float | None) -> float | None:
    if bks is None or abs(bks) <= 1e-12:
        return None
    return 100.0 * (bks - score) / bks


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    instance = parse_instance(job["instance_path"])
    protocol = str(job["protocol"])
    reference_budget = job.get("reference_runtime_seconds")
    effective_limit: float | None = None
    if reference_budget is not None:
        reference_budget = float(reference_budget)
        guard = (
            min(4.0, max(1.0, 0.02 * reference_budget))
            if job["algorithm"] == "kim_alns"
            else min(0.5, max(0.1, 0.002 * reference_budget))
        )
        effective_limit = max(0.0, reference_budget - guard)

    result = solve_kim_alns(
        instance,
        seed=int(job["seed"]),
        max_iterations=(
            int(job["iterations"])
            if protocol == "fixed_iterations"
            else 2_147_483_647
        ),
        time_limit_seconds=effective_limit,
        config=KimALNSConfig(),
    )
    solution = result.solution
    runtime = float(result.stats.runtime_seconds)
    completed_iterations = int(result.stats.completed_iterations)
    extras = {
        "position_updates": None,
        "swarm_rounds": None,
        "completed_heuristic_initialisations": None,
        "pool_size": result.stats.pool_size,
        "accepted_pool_updates": result.stats.accepted_pool_updates,
    }

    validation = validate_solution(solution, instance, build_distance_matrix(instance))
    bks = float(job["bks"]) if job.get("bks") is not None else None
    reward = float(solution.total_reward)
    within = (
        runtime <= float(reference_budget) + 1e-9
        if reference_budget is not None
        else None
    )
    return {
        "instance": job["instance"],
        "algorithm": job["algorithm"],
        "algorithm_label": ALGORITHM_LABELS[job["algorithm"]],
        "protocol": protocol,
        "seed": int(job["seed"]),
        "requested_iterations": (
            int(job["iterations"]) if protocol == "fixed_iterations" else None
        ),
        "completed_iterations": completed_iterations,
        "reference_runtime_seconds": reference_budget,
        "effective_time_limit_seconds": effective_limit,
        "runtime_seconds": runtime,
        "runtime_slack_seconds": (
            float(reference_budget) - runtime
            if reference_budget is not None
            else None
        ),
        "within_time_budget": within,
        "timeout": bool(result.stats.timed_out),
        "termination_reason": result.stats.termination_reason,
        "best_reward": reward,
        "bks": bks,
        "bks_gap_percent": gap_percent(reward, bks),
        "total_distance": float(solution.total_distance),
        "visited_node_count": len(solution.visited_nodes),
        "feasible": bool(validation["feasible"]),
        "violations": validation["violations"],
        **extras,
        "completed_at_utc": utc_now(),
    }


def summarize(
    rows: list[dict[str, Any]], expected_seeds: int, iterations: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["instance"])].append(row)
    per_instance: list[dict[str, Any]] = []
    for instance, runs in sorted(grouped.items()):
        rewards = [float(run["best_reward"]) for run in runs]
        best_reward = max(rewards)
        best_run = min(
            (run for run in runs if float(run["best_reward"]) == best_reward),
            key=lambda run: float(run["total_distance"]),
        )
        bks = float(best_run["bks"])
        mean_reward = mean(rewards)
        per_instance.append(
            {
                "instance": instance,
                "completed_seeds": len(runs),
                "expected_seeds": expected_seeds,
                "best_reward": best_reward,
                "mean_reward": mean_reward,
                "reward_std": pstdev(rewards),
                "best_seed": int(best_run["seed"]),
                "bks": bks,
                "gap_best_percent": gap_percent(best_reward, bks),
                "gap_mean_percent": gap_percent(mean_reward, bks),
                "mean_runtime_seconds": mean(float(run["runtime_seconds"]) for run in runs),
                "mean_completed_iterations": mean(
                    float(run["completed_iterations"]) for run in runs
                ),
                "all_fixed_iterations_complete": all(
                    int(run["completed_iterations"]) == iterations for run in runs
                ),
                "all_feasible": all(bool(run["feasible"]) for run in runs),
                "all_within_budget": all(
                    run["within_time_budget"] is not False for run in runs
                ),
                "timeouts": sum(bool(run["timeout"]) for run in runs),
            }
        )
    complete = [row for row in per_instance if row["completed_seeds"] == expected_seeds]
    best_gaps = [float(row["gap_best_percent"]) for row in complete]
    mean_gaps = [float(row["gap_mean_percent"]) for row in complete]
    aggregate = {
        "algorithm": rows[0]["algorithm"] if rows else None,
        "algorithm_label": rows[0]["algorithm_label"] if rows else None,
        "protocol": rows[0]["protocol"] if rows else None,
        "instances_reported": len(per_instance),
        "instances_complete": len(complete),
        "expected_seeds": expected_seeds,
        "runs_reported": len(rows),
        "all_runs_feasible": all(bool(row["feasible"]) for row in rows),
        "all_runs_within_budget": all(
            row["within_time_budget"] is not False for row in rows
        ),
        "all_fixed_iterations_complete": all(
            int(row["completed_iterations"]) == iterations for row in rows
        ),
        "timeouts": sum(bool(row["timeout"]) for row in rows),
        "mean_best_gap_percent": mean(best_gaps) if best_gaps else None,
        "median_best_gap_percent": median(best_gaps) if best_gaps else None,
        "mean_seed_gap_percent": mean(mean_gaps) if mean_gaps else None,
        "median_seed_gap_percent": median(mean_gaps) if mean_gaps else None,
        "bks_hits": sum(gap <= EPSILON for gap in best_gaps),
        "within_1_percent": sum(gap <= 1.0 + EPSILON for gap in best_gaps),
        "within_3_percent": sum(gap <= 3.0 + EPSILON for gap in best_gaps),
        "within_5_percent": sum(gap <= 5.0 + EPSILON for gap in best_gaps),
        "mean_runtime_seconds": (
            mean(float(row["runtime_seconds"]) for row in rows) if rows else None
        ),
        "mean_completed_iterations": (
            mean(float(row["completed_iterations"]) for row in rows) if rows else None
        ),
    }
    return per_instance, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=sorted(ALGORITHM_LABELS), required=True)
    parser.add_argument(
        "--protocol", choices=("fixed_iterations", "time_matched"), required=True
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--budgets", type=Path, default=DEFAULT_BUDGET_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument(
        "--force-rerun",
        nargs="*",
        default=[],
        metavar="INSTANCE:SEED",
        help="Append replacement runs for selected completed keys.",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.workers <= 0 or args.budget_scale <= 0:
        raise SystemExit("iterations, workers and budget-scale must be positive")

    instances = discover_instances(args.benchmark_root, dataset="dang")
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        instances = instances[: args.limit]
    elif len(instances) != 82:
        raise SystemExit(f"Expected 82 Dang instances, found {len(instances)}")
    seeds = list(dict.fromkeys(args.seeds))
    forced: set[tuple[str, int]] = set()
    for value in args.force_rerun:
        try:
            instance_name, seed_text = value.rsplit(":", 1)
            forced.add((instance_name.lower(), int(seed_text)))
        except ValueError as exc:
            raise SystemExit("--force-rerun values must be INSTANCE:SEED") from exc
    references = load_published_references(args.benchmark_root)
    budgets = read_budgets(args.budgets) if args.protocol == "time_matched" else {}
    configuration = {
        "algorithm": args.algorithm,
        "algorithm_label": ALGORITHM_LABELS[args.algorithm],
        "protocol": args.protocol,
        "benchmark_root": str(args.benchmark_root.resolve()),
        "dataset": "Dang2013-82",
        "instance_count": len(instances),
        "seeds": seeds,
        "iterations": args.iterations,
        "iteration_semantics": (
            "one complete ruin-improve-pool-update outer iteration "
            "(Kim et al. Algorithm 1)"
        ),
        "budget_source": (
            str(args.budgets.resolve()) if args.protocol == "time_matched" else None
        ),
        "budget_scale": args.budget_scale,
        "parameters": dataclass_values(
            KimALNSConfig()
        ),
        "implementation_level_reproduction": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration") != configuration:
            raise SystemExit("Output directory belongs to a different experiment")

    raw_path = args.output_dir / "raw_runs.jsonl"
    records = deduplicate(read_jsonl(raw_path))
    done = {key(row) for row in records}
    jobs: list[dict[str, Any]] = []
    for path in instances:
        instance_key = path.stem.lower()
        reference = references.get(instance_key)
        for seed in seeds:
            if (instance_key, seed) in done and (instance_key, seed) not in forced:
                continue
            jobs.append(
                {
                    "instance": path.stem,
                    "instance_path": str(path.resolve()),
                    "algorithm": args.algorithm,
                    "protocol": args.protocol,
                    "seed": seed,
                    "iterations": args.iterations,
                    "reference_runtime_seconds": (
                        budgets[instance_key] * args.budget_scale
                        if args.protocol == "time_matched"
                        else None
                    ),
                    "bks": reference.best_known if reference else None,
                }
            )

    total = len(instances) * len(seeds)
    manifest = {
        "configuration": configuration,
        "status": "running",
        "created_or_resumed_at_utc": utc_now(),
        "workers": args.workers,
        "completed_runs": len(records),
        "total_runs": total,
    }
    write_json(manifest_path, manifest)
    failures = 0
    with raw_path.open("a", encoding="utf-8", buffering=1) as append_handle:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
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
                    with (args.output_dir / "failures.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                else:
                    append_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    append_handle.flush()
                    records.append(row)
                    records = deduplicate(records)
                    write_csv(args.output_dir / "raw_runs.csv", records, RAW_FIELDS)
                    per_instance, aggregate = summarize(records, len(seeds), args.iterations)
                    write_csv(
                        args.output_dir / "summary.csv",
                        per_instance,
                        per_instance[0].keys() if per_instance else (),
                    )
                    write_json(args.output_dir / "aggregate.json", aggregate)
                    budget_text = (
                        f"/{row['reference_runtime_seconds']:.3f}s"
                        if row["reference_runtime_seconds"] is not None
                        else "s"
                    )
                    print(
                        f"[{len(records)}/{total}] {row['instance']} seed={row['seed']} "
                        f"reward={row['best_reward']:.6g} gap={row['bks_gap_percent']:.4f}% "
                        f"iter={row['completed_iterations']} "
                        f"t={row['runtime_seconds']:.3f}{budget_text}",
                        flush=True,
                    )

    records = deduplicate(read_jsonl(raw_path))
    per_instance, aggregate = summarize(records, len(seeds), args.iterations)
    write_csv(args.output_dir / "raw_runs.csv", records, RAW_FIELDS)
    write_csv(
        args.output_dir / "summary.csv",
        per_instance,
        per_instance[0].keys() if per_instance else (),
    )
    write_json(args.output_dir / "aggregate.json", aggregate)
    status = "complete" if len(records) == total and failures == 0 else "incomplete"
    manifest.update(
        {
            "status": status,
            "finished_at_utc": utc_now(),
            "completed_runs": len(records),
            "failed_runs_this_invocation": failures,
        }
    )
    write_json(manifest_path, manifest)
    print(f"saved={args.output_dir.resolve()} records={len(records)}/{total} status={status}")


if __name__ == "__main__":
    main()
