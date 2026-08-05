"""Run comparison baselines under per-instance time budgets.

Each budget is read from the proposed method's 2,500-iteration raw run with
the same instance and seed.  Results are appended immediately and the same
command resumes missing or previously over-budget jobs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from run_horizontal_baselines import (  # noqa: E402
    ALGORITHM_LABELS,
    build_summaries,
    deduplicate,
    job_key,
    progress,
    read_jsonl,
    utc_now,
    write_csv,
    write_json,
)
from top_alns.benchmarks.references import load_published_references  # noqa: E402
from top_alns.benchmarks.runner import discover_instances  # noqa: E402
from top_alns.comparison_baselines import solve_comparison_baseline  # noqa: E402
from top_alns.distance import build_distance_matrix  # noqa: E402
from top_alns.feasibility import validate_solution  # noqa: E402
from top_alns.open_source_baselines import solve_pyvrp_top  # noqa: E402
from top_alns.parser import parse_instance  # noqa: E402
ALGORITHMS = (
    "grasp",
    "ils",
    "vns",
    "pyvrp",
)
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runtime_budget_grasp_ils_vns_pyvrp"
MAX_SEARCH_ITERATIONS = 2_147_483_647
RAW_FIELDS = (
    "instance",
    "algorithm",
    "algorithm_label",
    "seed",
    "reference_iterations",
    "reference_runtime_seconds",
    "effective_time_limit_seconds",
    "runtime_seconds",
    "runtime_slack_seconds",
    "within_time_budget",
    "completed_iterations",
    "best_reward",
    "total_distance",
    "visited_node_count",
    "feasible",
    "completed_at_utc",
)


def load_budgets(
    paths: list[Path],
    seeds: list[int],
    reference_iterations: int,
    budget_mode: str,
) -> dict[tuple[str, int], float]:
    selected_seeds = set(seeds)
    budgets: dict[tuple[str, int], float] = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        configuration = report.get("configuration", {})
        if int(configuration.get("iterations", -1)) != reference_iterations:
            raise SystemExit(
                f"Reference {path} is not a {reference_iterations}-iteration run"
            )
        for row in report.get("raw_runs", []):
            if not bool(row.get("ejection", True)):
                continue
            seed = int(row["seed"])
            if seed not in selected_seeds:
                continue
            key = (str(row["instance"]).lower(), seed)
            value = float(row["runtime_seconds"])
            if value <= 0.0:
                raise SystemExit(f"Non-positive runtime for {key} in {path}")
            if key in budgets and abs(budgets[key] - value) > 1e-9:
                raise SystemExit(f"Conflicting runtime references for {key}")
            budgets[key] = value
    if budget_mode == "per_seed":
        return budgets

    by_instance: dict[str, list[float]] = {}
    for (instance, seed), value in budgets.items():
        if seed in selected_seeds:
            by_instance.setdefault(instance, []).append(value)
    averaged: dict[tuple[str, int], float] = {}
    for instance, values in by_instance.items():
        if len(values) != len(selected_seeds):
            continue
        instance_mean = sum(values) / len(values)
        for seed in selected_seeds:
            averaged[(instance, seed)] = instance_mean
    return averaged


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
    effective_limit = max(0.0, budget - safety)
    completed_iterations = 0

    start = perf_counter()
    if job["algorithm"] in {"grasp", "ils", "vns"}:
        progress_state = [0]

        def capture_progress(current: int, total: int) -> None:
            del total
            progress_state[0] = current

        solution = solve_comparison_baseline(
            instance,
            job["algorithm"],
            max_iterations=MAX_SEARCH_ITERATIONS,
            seed=job["seed"],
            progress_callback=capture_progress,
            time_limit_seconds=effective_limit,
        )
        completed_iterations = progress_state[0]
    elif job["algorithm"] == "pyvrp":
        solution, completed_iterations = solve_pyvrp_top(
            instance,
            seed=job["seed"],
            time_limit_seconds=effective_limit,
        )
    else:  # pragma: no cover - argparse validates names.
        raise ValueError(f"unsupported algorithm: {job['algorithm']}")
    runtime = perf_counter() - start

    matrix = build_distance_matrix(instance)
    validation = validate_solution(solution, instance, matrix)
    return {
        "instance": job["instance"],
        "algorithm": job["algorithm"],
        "algorithm_label": ALGORITHM_LABELS[job["algorithm"]],
        "seed": job["seed"],
        "reference_iterations": job["reference_iterations"],
        "reference_runtime_seconds": budget,
        "effective_time_limit_seconds": effective_limit,
        "runtime_seconds": runtime,
        "runtime_slack_seconds": budget - runtime,
        "within_time_budget": runtime <= budget,
        "completed_iterations": completed_iterations,
        "best_reward": float(solution.total_reward),
        "total_distance": float(solution.total_distance),
        "visited_node_count": len(solution.visited_nodes),
        "feasible": bool(validation["feasible"]),
        "completed_at_utc": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparison baselines with main-method time budgets."
    )
    parser.add_argument("--reference-json", nargs="+", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--reference-iterations", type=int, default=2500)
    parser.add_argument(
        "--budget-mode",
        choices=("per_seed", "instance_seed_mean"),
        default="per_seed",
        help=(
            "per_seed uses the paired seed runtime; instance_seed_mean gives "
            "all seeds the proposed method's mean runtime on that instance"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    parser.add_argument("--removal-rate", type=float, default=0.2)
    parser.add_argument("--minimum-removal-rate", type=float, default=0.05)
    parser.add_argument("--initial-temperature", type=float, default=20.0)
    parser.add_argument("--minimum-temperature", type=float, default=0.01)
    parser.add_argument("--safety-fraction", type=float, default=0.002)
    parser.add_argument("--minimum-safety-seconds", type=float, default=0.10)
    parser.add_argument("--maximum-safety-seconds", type=float, default=0.50)
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help="0 disables in-pool worker recycling",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Run at most this many missing jobs, then exit cleanly for a fresh pool",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithms = list(dict.fromkeys(args.algorithms))
    seeds = list(dict.fromkeys(args.seeds))
    if args.workers <= 0 or args.reference_iterations <= 0:
        raise SystemExit("workers and reference iterations must be positive")
    if args.max_tasks_per_child < 0:
        raise SystemExit("max tasks per child must be non-negative")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("batch size must be positive")

    benchmark_root = args.benchmark_root.resolve()
    instances = discover_instances(benchmark_root, dataset="dang", pattern="*.txt")
    if len(instances) != 82:
        raise SystemExit(f"Expected 82 Dang instances, found {len(instances)}")

    reference_paths = [path.resolve() for path in args.reference_json]
    budgets = load_budgets(
        reference_paths,
        seeds,
        args.reference_iterations,
        args.budget_mode,
    )
    required = {(path.stem.lower(), seed) for path in instances for seed in seeds}
    missing = sorted(required - set(budgets))
    if missing:
        preview = ", ".join(f"{name}/seed{seed}" for name, seed in missing[:10])
        raise SystemExit(f"Missing {len(missing)} time budgets: {preview}")

    configuration = {
        "protocol": (
            "per_instance_mean_seed_time_budget"
            if args.budget_mode == "instance_seed_mean"
            else "paired_per_instance_per_seed_time_budget"
        ),
        "budget_mode": args.budget_mode,
        "benchmark_root": str(benchmark_root),
        "instance_count": len(instances),
        "algorithms": algorithms,
        "seeds": seeds,
        "reference_iterations": args.reference_iterations,
        "reference_json": [str(path) for path in reference_paths],
        "workers": args.workers,
        "removal_rate": args.removal_rate,
        "minimum_removal_rate": args.minimum_removal_rate,
        "initial_temperature": args.initial_temperature,
        "minimum_temperature": args.minimum_temperature,
        "safety_fraction": args.safety_fraction,
        "minimum_safety_seconds": args.minimum_safety_seconds,
        "maximum_safety_seconds": args.maximum_safety_seconds,
        "max_tasks_per_child": args.max_tasks_per_child,
        "batch_size": args.batch_size,
        "pyvrp_version": "0.13.4",
        "over_budget_policy": "exclude_and_retry",
    }
    total = len(instances) * len(algorithms) * len(seeds)
    print(json.dumps({**configuration, "total_runs": total}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    raw_path = output_dir / "raw_runs.jsonl"
    failure_path = output_dir / "failures.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("configuration") != configuration:
            raise SystemExit("Output directory contains a different experiment")

    previous = deduplicate(read_jsonl(raw_path))
    valid_previous = [row for row in previous if bool(row.get("within_time_budget"))]
    completed = {job_key(row) for row in valid_previous}
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
                        "reference_iterations": args.reference_iterations,
                        "reference_runtime_seconds": budgets[(instance_path.stem.lower(), seed)],
                        "removal_rate": args.removal_rate,
                        "minimum_removal_rate": args.minimum_removal_rate,
                        "initial_temperature": args.initial_temperature,
                        "minimum_temperature": args.minimum_temperature,
                        "safety_fraction": args.safety_fraction,
                        "minimum_safety_seconds": args.minimum_safety_seconds,
                        "maximum_safety_seconds": args.maximum_safety_seconds,
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
    batch_jobs = jobs[: args.batch_size] if args.batch_size is not None else jobs
    batch_limited = len(batch_jobs) < len(jobs)
    failures = 0
    with raw_path.open("a", encoding="utf-8", buffering=1) as raw_handle:
        if batch_jobs:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                max_tasks_per_child=(
                    args.max_tasks_per_child or None
                ),
            ) as executor:
                future_to_job = {
                    executor.submit(run_one, job): job for job in batch_jobs
                }
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
    valid_records = [row for row in records if bool(row.get("within_time_budget"))]
    references = load_published_references(benchmark_root)
    summaries, aggregate = build_summaries(valid_records, references, len(seeds))
    write_csv(output_dir / "raw_runs.csv", records, RAW_FIELDS)
    if summaries:
        write_csv(output_dir / "summary.csv", summaries, summaries[0].keys())
    if aggregate:
        write_csv(output_dir / "aggregate.csv", aggregate, aggregate[0].keys())
    write_json(
        output_dir / "results.json",
        {
            "configuration": configuration,
            "raw_runs": records,
            "summary": summaries,
            "aggregate": aggregate,
        },
    )

    complete_count = len(valid_records)
    over_budget = sum(not bool(row.get("within_time_budget")) for row in records)
    manifest.update(
        {
            "status": "complete" if complete_count == total and failures == 0 else "incomplete",
            "finished_at_utc": utc_now(),
            "completed_runs": complete_count,
            "over_budget_records": over_budget,
            "failed_runs_this_invocation": failures,
        }
    )
    write_json(manifest_path, manifest)
    print(f"Completed within budget: {complete_count}/{total}; over-budget records: {over_budget}")
    print(f"Results: {output_dir}")
    if complete_count != total and not (batch_limited and failures == 0):
        raise SystemExit("Experiment incomplete; rerun the same command to retry missing/over-budget jobs")


if __name__ == "__main__":
    main()
