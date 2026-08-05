#!/usr/bin/env python
"""Run the reviewer-requested mechanism, ablation, and P/C experiments.

The suite uses the nine predefined sensitivity instances, two seeds, and the
paper's 2,500-iteration primary budget. Each completed run is written to its
own JSON file, so the command is safe to stop and resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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


DEFAULT_ITERATIONS = 2500
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
CONFIGS = (
    {
        "name": "density_p3_c2_l1",
        "ranking": "removal_density",
        "positions": 3,
        "ejections": 2,
        "passes": 1,
        "role": "reference_and_mechanism_logging",
    },
    {
        "name": "random_p3_c2_l1",
        "ranking": "random",
        "positions": 3,
        "ejections": 2,
        "passes": 1,
        "role": "removal_ranking_ablation",
    },
    {
        "name": "density_p3_c2_l0",
        "ranking": "removal_density",
        "positions": 3,
        "ejections": 2,
        "passes": 0,
        "role": "bounded_two_opt_ablation",
    },
    {
        "name": "density_p1_c2_l1",
        "ranking": "removal_density",
        "positions": 1,
        "ejections": 2,
        "passes": 1,
        "role": "P_sensitivity",
    },
    {
        "name": "density_p5_c2_l1",
        "ranking": "removal_density",
        "positions": 5,
        "ejections": 2,
        "passes": 1,
        "role": "P_sensitivity",
    },
    {
        "name": "density_p3_c1_l1",
        "ranking": "removal_density",
        "positions": 3,
        "ejections": 1,
        "passes": 1,
        "role": "C_sensitivity",
    },
    {
        "name": "density_p3_c3_l1",
        "ranking": "removal_density",
        "positions": 3,
        "ejections": 3,
        "passes": 1,
        "role": "C_sensitivity",
    },
)


def _job_key(job: dict) -> tuple[str, str, int]:
    return job["instance"], job["config"], int(job["seed"])


def _result_path(output_dir: Path, job: dict) -> Path:
    return output_dir / "runs" / (
        f"{job['instance']}__{job['config']}__seed{job['seed']}.json"
    )


def _progress_path(output_dir: Path, job: dict) -> Path:
    return output_dir / "progress" / (
        f"{job['instance']}__{job['config']}__seed{job['seed']}.json"
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _run_one(job: dict) -> dict:
    instance = parse_instance(job["path"])
    progress_path = _progress_path(Path(job["output_dir"]), job)
    record_every = max(25, int(job["iterations"]) // 100)

    def progress(iteration: int, total: int) -> None:
        if iteration % record_every == 0 or iteration == total:
            _atomic_json(
                progress_path,
                {"iteration": iteration, "total": total, "updated": time.time()},
            )

    solver = ALNSolver(
        max_iterations=job["iterations"],
        random_seed=job["seed"],
        enable_ejection=True,
        ejection_max_positions=job["positions"],
        ejection_max_nodes=job["ejections"],
        ejection_max_attempts=100,
        ejection_two_opt_passes=job["passes"],
        ejection_removal_ranking=job["ranking"],
        collect_ejection_telemetry=True,
    )
    started = time.perf_counter()
    solution = solver.solve(instance, progress_callback=progress)
    runtime = time.perf_counter() - started
    matrix = build_distance_matrix(instance)
    result = {
        "size": job["size"],
        "instance": job["instance"],
        "bks": job["bks"],
        "config": job["config"],
        "role": job["role"],
        "ranking": job["ranking"],
        "positions": job["positions"],
        "ejections": job["ejections"],
        "passes": job["passes"],
        "attempts": 100,
        "seed": job["seed"],
        "iterations": job["iterations"],
        "best_reward": solution.total_reward,
        "gap_percent": 100.0 * (job["bks"] - solution.total_reward) / job["bks"],
        "total_distance": solution.total_distance,
        "runtime_seconds": runtime,
        "feasible": check_solution_feasible(solution, instance, matrix),
        "routes": [route.node_ids for route in solution.routes],
        "search_diagnostics": solver.last_run_stats,
    }
    _atomic_json(_result_path(Path(job["output_dir"]), job), result)
    return result


def _load_completed(output_dir: Path) -> dict[tuple[str, str, int], dict]:
    completed: dict[tuple[str, str, int], dict] = {}
    for path in sorted((output_dir / "runs").glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        completed[_job_key(result)] = result
    return completed


def _telemetry(row: dict) -> dict:
    return row["search_diagnostics"]["ejection_telemetry"]


def _write_outputs(output_dir: Path, results: list[dict]) -> None:
    raw_fields = (
        "size", "instance", "bks", "config", "role", "ranking", "positions",
        "ejections", "passes", "attempts", "seed", "iterations", "best_reward",
        "gap_percent", "total_distance", "runtime_seconds", "feasible",
    )
    with (output_dir / "raw_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in raw_fields} for row in results)

    mechanism_fields = (
        *raw_fields,
        "repair_calls", "calls_with_blocked_customers", "blocked_customers",
        "attempted_blocked_customers", "successful_commits",
        "success_rate_per_attempted_customer", "success_rate_per_repair_call",
        "success_c0", "success_c1", "success_c2", "success_c3",
        "bounded_two_opt_trial_calls",
        "successful_commits_using_bounded_two_opt",
        "bounded_two_opt_share_of_successes",
        "mean_net_reward_gain_per_success", "net_reward_gain_sum",
        "new_best_candidates_with_successful_repair",
        "successful_commits_in_new_best_candidates", "module_seconds",
        "total_solve_seconds", "module_runtime_share",
    )
    with (output_dir / "mechanism_runs.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=mechanism_fields)
        writer.writeheader()
        for row in results:
            telemetry = _telemetry(row)
            counts = telemetry["successful_by_ejection_count"]
            writer.writerow(
                {
                    **{key: row.get(key) for key in raw_fields},
                    **{
                        key: telemetry.get(key)
                        for key in mechanism_fields
                        if key not in raw_fields and not key.startswith("success_c")
                    },
                    "success_c0": counts.get("0", 0),
                    "success_c1": counts.get("1", 0),
                    "success_c2": counts.get("2", 0),
                    "success_c3": counts.get("3", 0),
                }
            )

    summary_fields = (
        "config", "role", "ranking", "positions", "ejections", "passes",
        "run_count", "mean_reward", "std_reward", "mean_gap_percent",
        "std_gap_percent", "mean_runtime_seconds", "all_feasible",
        "blocked_customers", "attempted_blocked_customers", "successful_commits",
        "success_rate_per_attempted_customer", "success_c0", "success_c1",
        "success_c2", "success_c3", "bounded_two_opt_trial_calls",
        "successful_commits_using_bounded_two_opt",
        "bounded_two_opt_share_of_successes", "mean_net_reward_gain_per_success",
        "new_best_candidates_with_successful_repair",
        "successful_commits_in_new_best_candidates", "module_seconds",
        "total_solve_seconds", "module_runtime_share",
    )
    summaries: list[dict] = []
    for config in CONFIGS:
        rows = [row for row in results if row["config"] == config["name"]]
        if not rows:
            continue
        telemetry_rows = [_telemetry(row) for row in rows]
        attempts = sum(row["attempted_blocked_customers"] for row in telemetry_rows)
        successes = sum(row["successful_commits"] for row in telemetry_rows)
        two_opt_successes = sum(
            row["successful_commits_using_bounded_two_opt"] for row in telemetry_rows
        )
        module_seconds = sum(row["module_seconds"] for row in telemetry_rows)
        total_seconds = sum(row["total_solve_seconds"] for row in telemetry_rows)
        net_gain = sum(row["net_reward_gain_sum"] for row in telemetry_rows)
        counts = {
            str(c): sum(row["successful_by_ejection_count"].get(str(c), 0) for row in telemetry_rows)
            for c in range(4)
        }
        summaries.append(
            {
                "config": config["name"],
                "role": config["role"],
                "ranking": config["ranking"],
                "positions": config["positions"],
                "ejections": config["ejections"],
                "passes": config["passes"],
                "run_count": len(rows),
                "mean_reward": mean(row["best_reward"] for row in rows),
                "std_reward": pstdev(row["best_reward"] for row in rows),
                "mean_gap_percent": mean(row["gap_percent"] for row in rows),
                "std_gap_percent": pstdev(row["gap_percent"] for row in rows),
                "mean_runtime_seconds": mean(row["runtime_seconds"] for row in rows),
                "all_feasible": all(row["feasible"] for row in rows),
                "blocked_customers": sum(row["blocked_customers"] for row in telemetry_rows),
                "attempted_blocked_customers": attempts,
                "successful_commits": successes,
                "success_rate_per_attempted_customer": successes / attempts if attempts else 0.0,
                "success_c0": counts["0"], "success_c1": counts["1"],
                "success_c2": counts["2"], "success_c3": counts["3"],
                "bounded_two_opt_trial_calls": sum(row["bounded_two_opt_trial_calls"] for row in telemetry_rows),
                "successful_commits_using_bounded_two_opt": two_opt_successes,
                "bounded_two_opt_share_of_successes": two_opt_successes / successes if successes else 0.0,
                "mean_net_reward_gain_per_success": net_gain / successes if successes else 0.0,
                "new_best_candidates_with_successful_repair": sum(row["new_best_candidates_with_successful_repair"] for row in telemetry_rows),
                "successful_commits_in_new_best_candidates": sum(row["successful_commits_in_new_best_candidates"] for row in telemetry_rows),
                "module_seconds": module_seconds,
                "total_solve_seconds": total_seconds,
                "module_runtime_share": module_seconds / total_seconds if total_seconds else 0.0,
            }
        )
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in summary_fields} for row in summaries)

    cases: list[dict] = []
    for row in results:
        telemetry = _telemetry(row)
        for outcome_key in ("success_cases", "failure_cases"):
            for case in telemetry[outcome_key]:
                cases.append(
                    {
                        "size": row["size"], "instance": row["instance"],
                        "config": row["config"], "seed": row["seed"], **case,
                    }
                )
    _atomic_json(output_dir / "mechanism_cases.json", {"cases": cases})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--output-dir", default="outputs/reviewer_mechanism_9instances_20260727"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.iterations <= 0:
        raise SystemExit("workers and iterations must be positive")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
                            "size": size, "instance": instance,
                            "path": str(paths[instance]),
                            "bks": references[instance.lower()].best_known,
                            "seed": seed, "iterations": args.iterations,
                            "output_dir": str(output_dir),
                            "config": config["name"], "role": config["role"],
                            "ranking": config["ranking"],
                            "positions": config["positions"],
                            "ejections": config["ejections"],
                            "passes": config["passes"],
                        }
                    )
    manifest = {
        "purpose": "reviewer-requested mechanism evidence and low-cost ablations",
        "iterations": args.iterations, "seeds": list(SEEDS),
        "instances": INSTANCE_GROUPS, "configs": CONFIGS,
        "fixed": {"ejection_max_attempts": 100},
        "metric_definitions": {
            "success_rate": "successful commits / attempted blocked customers",
            "bounded_two_opt_share": "successful commits that invoked bounded 2-opt / successful commits",
            "direct_new_best": "candidate containing a successful repair commit that immediately updated the global best",
            "runtime_share": "timed compound repair module / total solve wall time",
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), **manifest}, ensure_ascii=False, indent=2))
        return

    completed = _load_completed(output_dir)
    pending = [job for job in jobs if _job_key(job) not in completed]
    print(
        f"total={len(jobs)}, completed={len(completed)}, pending={len(pending)}, "
        f"workers={args.workers}, iterations={args.iterations}", flush=True
    )
    failures: list[dict] = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_one, job): job for job in pending}
            for number, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    result = future.result()
                    completed[_job_key(result)] = result
                    print(
                        f"[{number}/{len(pending)}] {job['instance']} "
                        f"{job['config']} seed={job['seed']} complete",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append({"job": job, "error": repr(exc)})
                    print(f"[FAILED] {job['instance']} {job['config']} seed={job['seed']}: {exc!r}", flush=True)

    results = sorted(
        completed.values(),
        key=lambda row: (row["size"], row["instance"], row["config"], row["seed"]),
    )
    _write_outputs(output_dir, results)
    if failures:
        _atomic_json(output_dir / "failures.json", {"failures": failures})
        raise SystemExit(f"{len(failures)} jobs failed; rerun the command to resume")
    if len(results) != len(jobs):
        raise SystemExit(f"only {len(results)}/{len(jobs)} jobs completed")
    if not all(row["feasible"] for row in results):
        raise SystemExit("at least one completed solution is infeasible")
    print(f"Completed all {len(results)} jobs. Results: {output_dir}")


if __name__ == "__main__":
    main()
