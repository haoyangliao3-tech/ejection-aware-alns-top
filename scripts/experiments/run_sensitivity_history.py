#!/usr/bin/env python
"""Run the ejection sensitivity experiment and save the global best every 50 iterations.

This is a standalone experiment runner. It does not modify anything under src/
or tests/. It is Windows-safe and supports resuming completed jobs.

Run:
    python run_sensitivity_history.py --workers 12

Outputs are written to:
    outputs/sensitivity_history_5000/
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from top_alns.alns.solver import ALNSolver
from top_alns.benchmarks.references import load_published_references
from top_alns.benchmarks.runner import discover_instances
from top_alns.distance import build_distance_matrix
from top_alns.feasibility import check_solution_feasible
from top_alns.parser import parse_instance


ITERATIONS = 5000
RECORD_EVERY = 50
SEEDS = (0, 1)

INSTANCE_GROUPS = {
    "small": (
        "bier127_gen1_m3",
        "bier127_gen2_m3",
        "bier127_gen3_m3",
    ),
    "medium": (
        "pr299_gen1_m3",
        "pr299_gen2_m3",
        "pr299_gen3_m3",
    ),
    "large": (
        "rd400_gen1_m3",
        "rd400_gen2_m3",
        "rd400_gen3_m3",
    ),
}

# One-factor-at-a-time sensitivity around the default K=100, passes=1.
CONFIGS = (
    {"name": "off", "enable_ejection": False, "attempts": 0, "passes": 1},
    {"name": "k6_l1", "enable_ejection": True, "attempts": 6, "passes": 1},
    {"name": "k20_l1", "enable_ejection": True, "attempts": 20, "passes": 1},
    {"name": "k100_l1", "enable_ejection": True, "attempts": 100, "passes": 1},
    {"name": "k100_l3", "enable_ejection": True, "attempts": 100, "passes": 3},
    {"name": "k100_l5", "enable_ejection": True, "attempts": 100, "passes": 5},
    {"name": "k100_linf", "enable_ejection": True, "attempts": 100, "passes": None},
)


def _job_key(job: dict) -> tuple[str, str, int]:
    return job["instance"], job["config"], int(job["seed"])


def _trajectory_path(output_dir: Path, job: dict) -> Path:
    filename = f"{job['instance']}__{job['config']}__seed{job['seed']}.jsonl"
    return output_dir / "trajectories" / filename


def _run_one(job: dict) -> dict:
    """Run one job. The worker writes a trajectory point every 50 iterations."""
    output_dir = Path(job["output_dir"])
    trajectory_path = _trajectory_path(output_dir, job)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    instance = parse_instance(job["path"])
    history: list[dict] = []

    # Overwrite a partial trajectory when an unfinished job is resumed.
    handle = trajectory_path.open("w", encoding="utf-8", buffering=1)

    def progress(iteration: int, total: int) -> None:
        if iteration % RECORD_EVERY != 0 and iteration != total:
            return

        # ALNSolver calls progress directly from solve(), so its caller frame
        # contains the current global-best solution in the local variable best.
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        best = caller.f_locals.get("best") if caller is not None else None
        if best is None:
            raise RuntimeError("Could not read the solver's global-best solution")

        point = {
            "iteration": iteration,
            "best_reward": best.total_reward,
            "gap_percent": 100.0
            * (job["bks"] - best.total_reward)
            / job["bks"],
            "best_distance": best.total_distance,
        }
        history.append(point)
        handle.write(json.dumps(point, ensure_ascii=False) + "\n")
        handle.flush()

    solver = ALNSolver(
        max_iterations=ITERATIONS,
        random_seed=job["seed"],
        enable_ejection=job["enable_ejection"],
        ejection_max_positions=3,
        ejection_max_nodes=2,
        ejection_max_attempts=job["attempts"],
        ejection_two_opt_passes=job["passes"],
    )

    started = time.perf_counter()
    try:
        solution = solver.solve(instance, progress_callback=progress)
    finally:
        handle.close()
    runtime = time.perf_counter() - started

    matrix = build_distance_matrix(instance)
    expected_points = ITERATIONS // RECORD_EVERY + 1
    history_valid = (
        len(history) == expected_points
        and history[-1]["best_reward"] == solution.total_reward
        and all(
            history[index]["best_reward"]
            <= history[index + 1]["best_reward"]
            for index in range(len(history) - 1)
        )
    )

    return {
        "size": job["size"],
        "instance": job["instance"],
        "bks": job["bks"],
        "config": job["config"],
        "enable_ejection": job["enable_ejection"],
        "attempts": job["attempts"],
        "passes": job["passes"],
        "seed": job["seed"],
        "iterations": ITERATIONS,
        "record_every": RECORD_EVERY,
        "best_reward": solution.total_reward,
        "gap_percent": 100.0
        * (job["bks"] - solution.total_reward)
        / job["bks"],
        "total_distance": solution.total_distance,
        "runtime_seconds": runtime,
        "feasible": check_solution_feasible(solution, instance, matrix),
        "history_valid": history_valid,
        "trajectory_file": str(trajectory_path),
        "search_diagnostics": solver.last_run_stats,
    }


def _load_completed(path: Path) -> dict[tuple[str, str, int], dict]:
    completed: dict[tuple[str, str, int], dict] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result = json.loads(line)
            completed[_job_key(result)] = result
    return completed


def _read_last_saved_iteration(path: Path) -> int:
    """Read the newest complete trajectory record while a worker is writing."""
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return 0
    for line in reversed(lines):
        try:
            return int(json.loads(line)["iteration"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return 0


def _progress_monitor(
    jobs: list[dict], stop_event: threading.Event, started: float
) -> None:
    """Display aggregate real-time progress using the 50-iteration records."""
    total_iterations = len(jobs) * ITERATIONS
    bar_width = 36
    while not stop_event.is_set():
        saved_iterations = sum(
            min(
                ITERATIONS,
                _read_last_saved_iteration(
                    _trajectory_path(Path(job["output_dir"]), job)
                ),
            )
            for job in jobs
        )
        finished_jobs = sum(
            _read_last_saved_iteration(
                _trajectory_path(Path(job["output_dir"]), job)
            )
            >= ITERATIONS
            for job in jobs
        )
        fraction = (
            saved_iterations / total_iterations if total_iterations else 1.0
        )
        filled = min(bar_width, int(round(fraction * bar_width)))
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = time.perf_counter() - started
        print(
            f"\rOverall [{bar}] {fraction * 100:6.2f}% | "
            f"jobs {finished_jobs:3d}/{len(jobs)} | "
            f"saved iterations {saved_iterations:,}/{total_iterations:,} | "
            f"elapsed {elapsed / 60:.1f} min",
            end="",
            flush=True,
        )
        stop_event.wait(1.0)

    saved_iterations = sum(
        min(
            ITERATIONS,
            _read_last_saved_iteration(
                _trajectory_path(Path(job["output_dir"]), job)
            ),
        )
        for job in jobs
    )
    finished_jobs = sum(
        _read_last_saved_iteration(
            _trajectory_path(Path(job["output_dir"]), job)
        )
        >= ITERATIONS
        for job in jobs
    )
    fraction = saved_iterations / total_iterations if total_iterations else 1.0
    filled = min(bar_width, int(round(fraction * bar_width)))
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\rOverall [{bar}] {fraction * 100:6.2f}% | "
        f"jobs {finished_jobs:3d}/{len(jobs)} | "
        f"saved iterations {saved_iterations:,}/{total_iterations:,}",
        flush=True,
    )


def _write_csv_files(output_dir: Path, results: list[dict]) -> None:
    raw_fields = (
        "size",
        "instance",
        "bks",
        "config",
        "enable_ejection",
        "attempts",
        "passes",
        "seed",
        "iterations",
        "record_every",
        "best_reward",
        "gap_percent",
        "total_distance",
        "runtime_seconds",
        "feasible",
        "history_valid",
        "trajectory_file",
    )
    with (output_dir / "raw_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in raw_fields} for row in results)

    trajectory_fields = (
        "size",
        "instance",
        "bks",
        "config",
        "attempts",
        "passes",
        "seed",
        "iteration",
        "best_reward",
        "gap_percent",
        "best_distance",
    )
    with (output_dir / "trajectory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fields)
        writer.writeheader()
        for result in results:
            path = Path(result["trajectory_file"])
            for line in path.read_text(encoding="utf-8").splitlines():
                point = json.loads(line)
                writer.writerow(
                    {
                        "size": result["size"],
                        "instance": result["instance"],
                        "bks": result["bks"],
                        "config": result["config"],
                        "attempts": result["attempts"],
                        "passes": result["passes"],
                        "seed": result["seed"],
                        **point,
                    }
                )

    summary_fields = (
        "size",
        "instance",
        "bks",
        "config",
        "attempts",
        "passes",
        "mean_reward",
        "std_reward",
        "mean_gap_percent",
        "std_gap_percent",
        "mean_runtime_seconds",
        "all_feasible",
        "all_history_valid",
    )
    summary: list[dict] = []
    for size, names in INSTANCE_GROUPS.items():
        for instance in names:
            for config in CONFIGS:
                rows = [
                    row
                    for row in results
                    if row["instance"] == instance
                    and row["config"] == config["name"]
                ]
                if not rows:
                    continue
                summary.append(
                    {
                        "size": size,
                        "instance": instance,
                        "bks": rows[0]["bks"],
                        "config": config["name"],
                        "attempts": config["attempts"],
                        "passes": config["passes"],
                        "mean_reward": mean(row["best_reward"] for row in rows),
                        "std_reward": pstdev(row["best_reward"] for row in rows),
                        "mean_gap_percent": mean(
                            row["gap_percent"] for row in rows
                        ),
                        "std_gap_percent": pstdev(
                            row["gap_percent"] for row in rows
                        ),
                        "mean_runtime_seconds": mean(
                            row["runtime_seconds"] for row in rows
                        ),
                        "all_feasible": all(row["feasible"] for row in rows),
                        "all_history_valid": all(
                            row["history_valid"] for row in rows
                        ),
                    }
                )

    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TOP ejection sensitivity and save global best every 50 iterations."
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        default="outputs/sensitivity_history_5000",
        help="Existing completed jobs are resumed automatically.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment size without running it.",
    )
    args = parser.parse_args()

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)

    benchmark_root = PROJECT_ROOT / "benchmarks"
    paths = {
        path.stem: path.resolve()
        for path in discover_instances(benchmark_root, "dang", "*.txt")
    }
    references = load_published_references(benchmark_root)

    jobs: list[dict] = []
    for size, names in INSTANCE_GROUPS.items():
        for instance in names:
            for config in CONFIGS:
                for seed in SEEDS:
                    jobs.append(
                        {
                            "size": size,
                            "instance": instance,
                            "path": str(paths[instance]),
                            "bks": references[instance.lower()].best_known,
                            "config": config["name"],
                            "enable_ejection": config["enable_ejection"],
                            "attempts": config["attempts"],
                            "passes": config["passes"],
                            "seed": seed,
                            "output_dir": str(output_dir),
                        }
                    )

    if args.dry_run:
        print(f"instances={sum(len(x) for x in INSTANCE_GROUPS.values())}")
        print(f"configs={len(CONFIGS)}")
        print(f"seeds={list(SEEDS)}")
        print(f"jobs={len(jobs)}")
        print(f"iterations={ITERATIONS}, record_every={RECORD_EVERY}")
        print(f"output={output_dir}")
        return

    raw_jsonl = output_dir / "raw_runs.jsonl"
    completed = _load_completed(raw_jsonl)
    pending = [job for job in jobs if _job_key(job) not in completed]

    manifest = {
        "iterations": ITERATIONS,
        "record_every": RECORD_EVERY,
        "seeds": list(SEEDS),
        "workers": args.workers,
        "instances": INSTANCE_GROUPS,
        "configs": CONFIGS,
        "fixed": {"ejection_max_positions": 3, "ejection_max_nodes": 2},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"total={len(jobs)}, completed={len(completed)}, "
        f"pending={len(pending)}, workers={args.workers}"
    )

    failures: list[dict] = []
    if pending:
        started = time.perf_counter()
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_progress_monitor,
            args=(jobs, stop_event, started),
            daemon=True,
        )
        monitor.start()
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_run_one, job): job for job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append({"job": job, "error": repr(exc)})
                        continue

                    completed[_job_key(result)] = result
                    with raw_jsonl.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(result, ensure_ascii=False) + "\n"
                        )
        finally:
            stop_event.set()
            monitor.join()

        for failure in failures:
            job = failure["job"]
            print(
                f"[FAILED] {job['instance']} {job['config']} "
                f"seed={job['seed']}: {failure['error']}"
            )

    results = sorted(
        completed.values(),
        key=lambda row: (row["size"], row["instance"], row["config"], row["seed"]),
    )
    _write_csv_files(output_dir, results)

    if failures:
        (output_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit(
            f"{len(failures)} jobs failed. Re-run the same command to resume."
        )

    if len(results) != len(jobs):
        raise SystemExit(
            f"Only {len(results)}/{len(jobs)} jobs completed. Re-run to resume."
        )
    if not all(row["feasible"] and row["history_valid"] for row in results):
        raise SystemExit("A feasibility or trajectory validation failed.")

    print(f"Completed all {len(results)} jobs.")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
