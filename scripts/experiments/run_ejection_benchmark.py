#!/usr/bin/env python
"""Generalization benchmark for the ejection repair operator.

For every selected instance this runs the *attention* ALNS twice — with the
compound insert-with-ejection move ON (the improved method) and OFF (the
pre-existing baseline) — across several seeds, in parallel, and reports the gap
to the published Best-Known Solution (BKS). Running the same algorithm with the
operator toggled isolates the operator's contribution and shows whether the gain
generalizes beyond a single instance.

Run it from the top_alns project directory (no install needed):

    # 12 Dang instances, 1000 iters, 3 seeds, compare ON vs OFF
    python run_ejection_benchmark.py --dataset dang --limit 12 --iterations 1000 --seeds 0 1 2

    # specific instances, more iters/seeds
    python run_ejection_benchmark.py --instances rd400_gen2_m3 kroA150_gen2_m2 \
        --iterations 2000 --seeds 0 1 2 3

    # whole Dang set (long!) using 14 workers, ejection ON only
    python run_ejection_benchmark.py --dataset dang --iterations 2000 \
        --seeds 0 1 2 --workers 14 --mode on

    # filter by name pattern (e.g. only rd400 family)
    python run_ejection_benchmark.py --dataset dang --pattern "rd400*" --iterations 2000 --seeds 0 1 2

Results (per-instance CSV + full JSON) are written to outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

# --- make the package importable without an install -----------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from top_alns.benchmarks.references import load_published_references  # noqa: E402
from top_alns.benchmarks.runner import discover_instances  # noqa: E402
from top_alns.distance import build_distance_matrix  # noqa: E402
from top_alns.feasibility import check_solution_feasible  # noqa: E402
from top_alns.parser import parse_instance  # noqa: E402
from top_alns.alns.solver import ALNSolver  # noqa: E402

BENCHMARK_ROOT = "benchmarks"


def _run_one(job: dict) -> dict:
    """Solve one (instance, ejection flag, seed) job. Runs in a worker process."""
    instance = parse_instance(job["path"])
    solver = ALNSolver(
        max_iterations=job["iterations"],
        random_seed=job["seed"],
        enable_ejection=job["ejection"],
    )
    start = time.perf_counter()
    solution = solver.solve(instance)
    runtime = time.perf_counter() - start
    matrix = build_distance_matrix(instance)
    feasible = check_solution_feasible(solution, instance, matrix)
    return {
        "instance": job["instance"],
        "ejection": job["ejection"],
        "seed": job["seed"],
        "reward": solution.total_reward,
        "distance": solution.total_distance,
        "feasible": feasible,
        "runtime_seconds": runtime,
    }


def _select_instances(args) -> list[Path]:
    root = _PROJECT_ROOT / BENCHMARK_ROOT
    instances = discover_instances(root, args.dataset, args.pattern)
    if args.instances:
        wanted = {name.lower() for name in args.instances}
        instances = [p for p in instances if p.stem.lower() in wanted]
        missing = wanted - {p.stem.lower() for p in instances}
        if missing:
            print(f"[warn] instances not found: {sorted(missing)}")
    if args.limit is not None:
        instances = instances[: args.limit]
    return instances


def _gap(reward: float, bks: float | None) -> float | None:
    if not bks:
        return None
    return 100.0 * (bks - reward) / bks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the ejection repair operator (ON vs OFF) vs BKS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=("all", "chao", "dang"), default="dang"
    )
    parser.add_argument(
        "--pattern", default="*.txt", help='name filter, e.g. "rd400*"'
    )
    parser.add_argument(
        "--instances", nargs="+", help="explicit instance stems to run"
    )
    parser.add_argument(
        "--limit", type=int, help="use only the first N discovered instances"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--mode",
        choices=("compare", "on", "off"),
        default="compare",
        help="compare = run ejection ON and OFF; on/off = only that variant",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="parallel worker processes (<= CPU cores)",
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    instances = _select_instances(args)
    if not instances:
        print("No instances matched. Check --dataset/--pattern/--instances.")
        return
    references = load_published_references(_PROJECT_ROOT / BENCHMARK_ROOT)
    flags = {"compare": [True, False], "on": [True], "off": [False]}[args.mode]

    jobs = [
        {
            "instance": path.stem,
            "path": str(path),
            "ejection": flag,
            "seed": int(seed),
            "iterations": args.iterations,
        }
        for path in instances
        for flag in flags
        for seed in args.seeds
    ]
    print(
        f"Instances={len(instances)}  seeds={args.seeds}  iters={args.iterations}"
        f"  mode={args.mode}  jobs={len(jobs)}  workers={args.workers}"
    )
    print("-" * 72)

    raw: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            raw.append(result)
            done += 1
            tag = "ON " if result["ejection"] else "OFF"
            print(
                f"[{done:3d}/{len(jobs)}] {result['instance']:20s} "
                f"eject={tag} seed={result['seed']} "
                f"reward={result['reward']:.0f} "
                f"t={result['runtime_seconds']:.0f}s",
                flush=True,
            )

    # --- aggregate per instance x variant ---------------------------------
    rows: list[dict] = []
    for path in instances:
        stem = path.stem
        ref = references.get(stem.lower())
        bks = ref.best_known if ref else None
        row: dict = {"instance": stem, "bks": bks}
        for flag in flags:
            key = "on" if flag else "off"
            rewards = [
                r["reward"]
                for r in raw
                if r["instance"] == stem and r["ejection"] == flag
            ]
            if not rewards:
                continue
            best = max(rewards)
            row[f"best_{key}"] = best
            row[f"mean_{key}"] = mean(rewards)
            row[f"gap_best_{key}"] = _gap(best, bks)
            row[f"gap_mean_{key}"] = _gap(mean(rewards), bks)
        rows.append(row)

    # --- print summary table ----------------------------------------------
    print("\n" + "=" * 72)
    print("Per-instance best reward and gap-to-BKS")
    print("=" * 72)
    header = f"{'instance':20s} {'BKS':>7s}"
    if True in flags:
        header += f" {'best_ON':>8s} {'gapON%':>7s}"
    if False in flags:
        header += f" {'best_OFF':>8s} {'gapOFF%':>7s}"
    if len(flags) == 2:
        header += f" {'dReward':>8s}"
    print(header)
    for row in rows:
        line = f"{row['instance']:20s} {str(row['bks'] or '-'):>7s}"
        if True in flags:
            g = row.get("gap_best_on")
            line += f" {row.get('best_on', 0):8.0f} {(f'{g:.2f}' if g is not None else '-'):>7s}"
        if False in flags:
            g = row.get("gap_best_off")
            line += f" {row.get('best_off', 0):8.0f} {(f'{g:.2f}' if g is not None else '-'):>7s}"
        if len(flags) == 2:
            line += f" {row.get('best_on', 0) - row.get('best_off', 0):+8.0f}"
        print(line)

    if len(flags) == 2:
        gaps_on = [r["gap_mean_on"] for r in rows if r.get("gap_mean_on") is not None]
        gaps_off = [r["gap_mean_off"] for r in rows if r.get("gap_mean_off") is not None]
        wins = sum(
            1
            for r in rows
            if r.get("best_on", 0) > r.get("best_off", 0) + 1e-9
        )
        ties = sum(
            1
            for r in rows
            if abs(r.get("best_on", 0) - r.get("best_off", 0)) <= 1e-9
        )
        print("-" * 72)
        if gaps_on and gaps_off:
            print(
                f"mean gap-to-BKS (mean reward):  ON={mean(gaps_on):.2f}%  "
                f"OFF={mean(gaps_off):.2f}%  improvement={mean(gaps_off) - mean(gaps_on):.2f} pts"
            )
        print(
            f"ejection ON best-reward: wins {wins}, ties {ties}, "
            f"losses {len(rows) - wins - ties}  (of {len(rows)} instances)"
        )

    # --- save CSV + JSON --------------------------------------------------
    out_dir = _PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ejection_benchmark_{stamp}.json"
    csv_path = out_dir / f"ejection_benchmark_{stamp}.csv"
    payload = {
        "configuration": {
            "dataset": args.dataset,
            "pattern": args.pattern,
            "instances": [p.stem for p in instances],
            "iterations": args.iterations,
            "seeds": args.seeds,
            "mode": args.mode,
        },
        "per_instance": rows,
        "raw_runs": raw,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print("-" * 72)
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
