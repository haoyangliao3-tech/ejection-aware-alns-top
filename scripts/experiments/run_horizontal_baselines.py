"""Run reproducible horizontal baselines on the 82 Dang TOP instances.

This script deliberately reuses the existing, independent solver modules and
does not modify the proposed Ejection-Aware ALNS implementation.  Each run is
saved immediately, so an interrupted experiment can be resumed safely by
executing the same command again.
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
from time import perf_counter
from typing import Any, Iterable

# Prevent each worker process from creating its own BLAS thread pool.
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
from top_alns.comparison_baselines import solve_comparison_baseline
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import validate_solution
from top_alns.parser import parse_instance
ALGORITHMS = ("grasp", "ils", "vns")
ALGORITHM_LABELS = {
    "grasp": "GRASP",
    "ils": "Iterated Local Search",
    "vns": "Variable Neighbourhood Search",
    "pyvrp": "PyVRP",
}
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fixed_iteration_grasp_ils_vns"
RAW_FIELDS = (
    "instance",
    "algorithm",
    "algorithm_label",
    "seed",
    "iterations",
    "best_reward",
    "total_distance",
    "visited_node_count",
    "feasible",
    "runtime_seconds",
    "completed_at_utc",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["instance"]).lower(),
        str(record["algorithm"]),
        int(record["seed"]),
    )


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point; it must remain at module scope for Windows."""
    instance = parse_instance(job["instance_path"])
    start = perf_counter()
    solution = solve_comparison_baseline(
        instance,
        job["algorithm"],
        max_iterations=job["iterations"],
        seed=job["seed"],
    )
    runtime = perf_counter() - start
    validation = validate_solution(
        solution, instance, build_distance_matrix(instance)
    )
    result = {
        "best_reward": solution.total_reward,
        "total_distance": solution.total_distance,
        "visited_node_count": len(solution.visited_nodes),
        "feasible": validation["feasible"],
        "runtime_seconds": runtime,
    }
    return {
        "instance": job["instance"],
        "algorithm": job["algorithm"],
        "algorithm_label": ALGORITHM_LABELS[job["algorithm"]],
        "seed": job["seed"],
        "iterations": job["iterations"],
        "best_reward": float(result["best_reward"]),
        "total_distance": float(result["total_distance"]),
        "visited_node_count": int(result["visited_node_count"]),
        "feasible": bool(result["feasible"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "completed_at_utc": utc_now(),
    }


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
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}; "
                    "do not edit raw_runs.jsonl while the experiment is running."
                ) from exc
    return records


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        by_key[job_key(record)] = record
    return sorted(
        by_key.values(),
        key=lambda row: (
            str(row["instance"]).lower(),
            str(row["algorithm"]),
            int(row["seed"]),
        ),
    )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gap_percent(score: float, bks: float | None) -> float | None:
    if bks is None or bks == 0:
        return None
    return 100.0 * (bks - score) / bks


def build_summaries(
    records: list[dict[str, Any]],
    references: dict[str, Any],
    expected_seeds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["instance"], record["algorithm"])].append(record)

    summaries: list[dict[str, Any]] = []
    for (instance, algorithm), runs in sorted(
        grouped.items(), key=lambda item: (item[0][0].lower(), item[0][1])
    ):
        rewards = [float(run["best_reward"]) for run in runs]
        best_reward = max(rewards)
        best_candidates = [
            run for run in runs if float(run["best_reward"]) == best_reward
        ]
        best_run = min(best_candidates, key=lambda run: float(run["total_distance"]))
        reference = references.get(instance.lower())
        bks = float(reference.best_known) if reference and reference.best_known is not None else None
        summaries.append(
            {
                "instance": instance,
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "completed_seeds": len(runs),
                "expected_seeds": expected_seeds,
                "best_reward": best_reward,
                "mean_reward": mean(rewards),
                "reward_std": pstdev(rewards),
                "best_seed": int(best_run["seed"]),
                "best_total_distance": float(best_run["total_distance"]),
                "mean_runtime_seconds": mean(float(run["runtime_seconds"]) for run in runs),
                "all_feasible": all(bool(run["feasible"]) for run in runs),
                "best_known": bks,
                "gap_best_percent": gap_percent(best_reward, bks),
                "gap_mean_percent": gap_percent(mean(rewards), bks),
            }
        )

    aggregate: list[dict[str, Any]] = []
    for algorithm in sorted({row["algorithm"] for row in summaries}):
        rows = [row for row in summaries if row["algorithm"] == algorithm]
        complete = [row for row in rows if row["completed_seeds"] == expected_seeds]
        gaps_best = [float(row["gap_best_percent"]) for row in complete if row["gap_best_percent"] is not None]
        gaps_mean = [float(row["gap_mean_percent"]) for row in complete if row["gap_mean_percent"] is not None]
        aggregate.append(
            {
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "instances_reported": len(rows),
                "instances_complete": len(complete),
                "all_runs_feasible": all(bool(row["all_feasible"]) for row in rows),
                "mean_best_gap_percent": mean(gaps_best) if gaps_best else None,
                "median_best_gap_percent": median(gaps_best) if gaps_best else None,
                "mean_gap_percent": mean(gaps_mean) if gaps_mean else None,
                "median_gap_percent": median(gaps_mean) if gaps_mean else None,
                "bks_hits": sum(gap <= 1e-9 for gap in gaps_best),
                "within_1_percent": sum(gap <= 1.0 for gap in gaps_best),
                "within_3_percent": sum(gap <= 3.0 for gap in gaps_best),
                "within_5_percent": sum(gap <= 5.0 for gap in gaps_best),
                "mean_runtime_seconds_per_seed": mean(
                    float(row["mean_runtime_seconds"]) for row in complete
                ) if complete else None,
            }
        )
    return summaries, aggregate


def progress(iterator: Iterable[Any], total: int, initial: int) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        return tqdm(
            iterator,
            total=total,
            initial=initial,
            desc="Horizontal baselines",
            unit="run",
            dynamic_ncols=True,
        )
    except ImportError:
        def fallback() -> Iterable[Any]:
            completed = initial
            for item in iterator:
                completed += 1
                print(f"\rHorizontal baselines: {completed}/{total}", end="", flush=True)
                yield item
            print()

        return fallback()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable GRASP, ILS and VNS baselines on Dang-82."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--removal-rate", type=float, default=0.2)
    parser.add_argument("--initial-temperature", type=float, default=20.0)
    parser.add_argument("--minimum-temperature", type=float, default=0.01)
    parser.add_argument("--limit", type=int, help="Use only the first N instances (smoke tests).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without solving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not args.seeds:
        raise SystemExit("At least one seed is required")

    benchmark_root = args.benchmark_root.resolve()
    instances = discover_instances(benchmark_root, dataset="dang", pattern="*.txt")
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        instances = instances[: args.limit]
    elif len(instances) != 82:
        raise SystemExit(
            f"Expected exactly 82 Dang instances, found {len(instances)} at {benchmark_root}"
        )

    algorithms = list(dict.fromkeys(args.algorithms))
    seeds = list(dict.fromkeys(args.seeds))
    configuration = {
        "benchmark_root": str(benchmark_root),
        "dataset": "Dang2013",
        "instance_count": len(instances),
        "algorithms": algorithms,
        "seeds": seeds,
        "iterations": args.iterations,
        "removal_rate": args.removal_rate,
        "initial_temperature": args.initial_temperature,
        "minimum_temperature": args.minimum_temperature,
        "iteration_semantics": {
            "grasp": "one randomized constructive insertion or restart",
            "ils": "one perturbation-repair-local-search iteration",
            "vns": "one neighbourhood perturbation-repair iteration",
        },
    }
    total = len(instances) * len(algorithms) * len(seeds)
    print(json.dumps({**configuration, "workers": args.workers, "total_runs": total}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    raw_path = output_dir / "raw_runs.jsonl"
    failure_path = output_dir / "failures.jsonl"
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("configuration") != configuration:
            raise SystemExit(
                "Output directory contains a different experiment. Choose a new "
                "--output-dir, or rerun with the original configuration."
            )

    previous = deduplicate(read_jsonl(raw_path))
    completed = {job_key(record) for record in previous}
    jobs: list[dict[str, Any]] = []
    for instance_path in instances:
        for algorithm in algorithms:
            for seed in seeds:
                key = (instance_path.stem.lower(), algorithm, seed)
                if key in completed:
                    continue
                jobs.append(
                    {
                        "instance": instance_path.stem,
                        "instance_path": str(instance_path.resolve()),
                        "algorithm": algorithm,
                        "seed": seed,
                        "iterations": args.iterations,
                        "removal_rate": args.removal_rate,
                        "initial_temperature": args.initial_temperature,
                        "minimum_temperature": args.minimum_temperature,
                    }
                )

    manifest = {
        "configuration": configuration,
        "status": "running",
        "started_or_resumed_at_utc": utc_now(),
        "completed_runs": len(completed),
        "total_runs": total,
    }
    write_json(manifest_path, manifest)
    failures = 0
    with raw_path.open("a", encoding="utf-8", buffering=1) as raw_handle:
        if jobs:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_job = {executor.submit(run_one, job): job for job in jobs}
                for future in progress(as_completed(future_to_job), total, len(completed)):
                    job = future_to_job[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        failures += 1
                        failure = {
                            "instance": job["instance"],
                            "algorithm": job["algorithm"],
                            "seed": job["seed"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_at_utc": utc_now(),
                        }
                        with failure_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    else:
                        raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        raw_handle.flush()

    records = deduplicate(read_jsonl(raw_path))
    references = load_published_references(benchmark_root)
    summaries, aggregate = build_summaries(records, references, len(seeds))
    write_csv(output_dir / "raw_runs.csv", records, RAW_FIELDS)
    summary_fields = list(summaries[0]) if summaries else []
    aggregate_fields = list(aggregate[0]) if aggregate else []
    if summary_fields:
        write_csv(output_dir / "summary.csv", summaries, summary_fields)
    if aggregate_fields:
        write_csv(output_dir / "aggregate.csv", aggregate, aggregate_fields)
    write_json(
        output_dir / "results.json",
        {"configuration": configuration, "raw_runs": records, "summary": summaries, "aggregate": aggregate},
    )

    complete_count = len(records)
    manifest.update(
        {
            "status": "complete" if complete_count == total and failures == 0 else "incomplete",
            "finished_at_utc": utc_now(),
            "completed_runs": complete_count,
            "failed_runs_this_invocation": failures,
        }
    )
    write_json(manifest_path, manifest)
    print(f"Completed {complete_count}/{total}; failures this run: {failures}")
    print(f"Results: {output_dir}")
    if complete_count != total:
        raise SystemExit("Experiment is incomplete. Run the same command again to retry missing jobs.")


if __name__ == "__main__":
    main()
