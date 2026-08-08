"""Build the public Dang+Chao workers=2 result bundle from raw run logs.

The script deliberately keeps fixed-iteration, symmetric wall-clock, and
runtime-budget comparator results in separate tables. Benchmark files are
hashed into a manifest but are not copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ALGORITHM_LABELS = {
    "on": "Ejection ON",
    "off": "Ejection OFF",
    "grasp": "GRASP",
    "ils": "ILS",
    "vns": "VNS",
    "pyvrp": "PyVRP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dang-root", type=Path, required=True)
    parser.add_argument("--chao-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "results" / "normalized_results.json",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def gap(bks: float, reward: float) -> float:
    return 100.0 * (float(bks) - float(reward)) / float(bks)


def records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def normalize_fixed_on_off(name: str, root: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, Path]]:
    paths: dict[str, Path] = {}
    if name == "Dang":
        source = next((root / "02_fixed_2500_ejection_on_off").glob("ejection_benchmark_*.json"))
        obj = json.loads(source.read_text(encoding="utf-8"))
        frame = pd.DataFrame(obj["raw_runs"])
        bks_map = {row["instance"]: float(row["bks"]) for row in obj["per_instance"]}
        frame["bks"] = frame["instance"].map(bks_map)
        frame["iterations"] = 2500
        frame["completed_iterations"] = 2500
        frame["visited_node_count"] = None
        frame["source"] = f"02_fixed_2500_ejection_on_off/{source.name}"
    else:
        source = root / "02_fixed_2500_ejection_on_off" / "runs.csv"
        frame = read_csv(source)
        bks_map = frame.groupby("instance")["bks"].first().astype(float).to_dict()
        frame["completed_iterations"] = frame["iterations"]
        frame["distance"] = frame["total_distance"]
        frame["visited_node_count"] = None
        frame["source"] = "02_fixed_2500_ejection_on_off/runs.csv"
        for row in frame[["instance", "instance_path"]].drop_duplicates().itertuples(index=False):
            paths[str(row.instance)] = Path(str(row.instance_path))

    frame["algorithm"] = np.where(frame["ejection"].astype(bool), "on", "off")
    frame["bks_gap_percent"] = [gap(b, r) for b, r in zip(frame["bks"], frame["reward"])]
    out = pd.DataFrame({
        "Protocol": "Fixed iteration",
        "Benchmark": name,
        "Algorithm": frame["algorithm"].map(ALGORITHM_LABELS),
        "Instance": frame["instance"],
        "Seed": frame["seed"].astype(int),
        "Iteration_Limit": frame["iterations"].astype(int),
        "Completed_Iterations": frame["completed_iterations"].astype(int),
        "Reward": frame["reward"].astype(float),
        "BKS": frame["bks"].astype(float),
        "BKS_Gap_Percent": frame["bks_gap_percent"].astype(float),
        "Total_Distance": frame["distance"].astype(float),
        "Visited_Customers": frame["visited_node_count"],
        "Runtime_Seconds": frame["runtime_seconds"].astype(float),
        "Feasible": frame["feasible"].astype(bool),
        "Batch_Workers": 2,
        "Source_Record": frame["source"],
    })
    return out, bks_map, paths


def normalize_fixed_baselines(name: str, root: Path, bks_map: dict[str, float]) -> pd.DataFrame:
    frame = read_csv(root / "03_fixed_2500_baselines" / "raw_runs.csv")
    bks = frame["instance"].map(bks_map).astype(float)
    return pd.DataFrame({
        "Protocol": "Fixed iteration",
        "Benchmark": name,
        "Algorithm": frame["algorithm"].map(ALGORITHM_LABELS),
        "Instance": frame["instance"],
        "Seed": frame["seed"].astype(int),
        "Iteration_Limit": frame["iterations"].astype(int),
        "Completed_Iterations": frame["iterations"].astype(int),
        "Reward": frame["best_reward"].astype(float),
        "BKS": bks,
        "BKS_Gap_Percent": [gap(b, r) for b, r in zip(bks, frame["best_reward"])],
        "Total_Distance": frame["total_distance"].astype(float),
        "Visited_Customers": frame["visited_node_count"].astype(int),
        "Runtime_Seconds": frame["runtime_seconds"].astype(float),
        "Feasible": frame["feasible"].astype(bool),
        "Batch_Workers": 2,
        "Source_Record": "03_fixed_2500_baselines/raw_runs.csv",
    })


def normalize_symmetric(name: str, root: Path) -> pd.DataFrame:
    frames = []
    for arm in ("on", "off"):
        frame = read_csv(root / "04_symmetric_wall_clock_ejection_on_off" / arm / "raw_runs.csv")
        frames.append(pd.DataFrame({
            "Protocol": "Symmetric wall clock",
            "Benchmark": name,
            "Algorithm": ALGORITHM_LABELS[arm],
            "Instance": frame["instance"],
            "Seed": frame["seed"].astype(int),
            "Reference_Iterations": frame["reference_iterations"].astype(int),
            "Budget_Seconds": frame["reference_runtime_seconds"].astype(float),
            "Effective_Time_Limit_Seconds": frame["effective_time_limit_seconds"].astype(float),
            "Completed_Iterations": frame["completed_iterations"].astype(int),
            "Reward": frame["reward"].astype(float),
            "BKS": frame["bks"].astype(float),
            "BKS_Gap_Percent": frame["bks_gap_percent"].astype(float),
            "Total_Distance": frame["distance"].astype(float),
            "Visited_Customers": frame["visited_node_count"].astype(int),
            "Runtime_Seconds": frame["runtime_seconds"].astype(float),
            "Runtime_Slack_Seconds": frame["runtime_slack_seconds"].astype(float),
            "Within_Budget": frame["within_time_budget"].astype(bool),
            "Timed_Out": frame["timed_out"].astype(bool),
            "Stop_Status": frame["timeout_status"],
            "Feasible": frame["feasible"].astype(bool),
            "Batch_Workers": 2,
            "Source_Record": f"04_symmetric_wall_clock_ejection_on_off/{arm}/raw_runs.csv",
        }))
    return pd.concat(frames, ignore_index=True)


def normalize_runtime_baselines(name: str, root: Path, bks_map: dict[str, float]) -> pd.DataFrame:
    frame = read_csv(root / "05_runtime_budget_baselines" / "raw_runs.csv")
    bks = frame["instance"].map(bks_map).astype(float)
    return pd.DataFrame({
        "Protocol": "Runtime budget baseline",
        "Benchmark": name,
        "Algorithm": frame["algorithm"].map(ALGORITHM_LABELS),
        "Instance": frame["instance"],
        "Seed": frame["seed"].astype(int),
        "Reference_Iterations": frame["reference_iterations"].astype(int),
        "Budget_Seconds": frame["reference_runtime_seconds"].astype(float),
        "Effective_Time_Limit_Seconds": frame["effective_time_limit_seconds"].astype(float),
        "Completed_Iterations": frame["completed_iterations"].astype(int),
        "Reward": frame["best_reward"].astype(float),
        "BKS": bks,
        "BKS_Gap_Percent": [gap(b, r) for b, r in zip(bks, frame["best_reward"])],
        "Total_Distance": frame["total_distance"].astype(float),
        "Visited_Customers": frame["visited_node_count"].astype(int),
        "Runtime_Seconds": frame["runtime_seconds"].astype(float),
        "Runtime_Slack_Seconds": frame["runtime_slack_seconds"].astype(float),
        "Within_Budget": frame["within_time_budget"].astype(bool),
        "Feasible": frame["feasible"].astype(bool),
        "Batch_Workers": 2,
        "Source_Record": "05_runtime_budget_baselines/raw_runs.csv",
    })


def normalize_focused(name: str, root: Path, stage: str, filename: str) -> pd.DataFrame:
    frame = read_csv(root / stage / filename)
    frame.insert(0, "Benchmark", name)
    frame["Batch_Workers"] = 2
    frame["Source_Record"] = f"{stage}/{filename}"
    for column in ("trajectory_file",):
        if column in frame:
            frame[column] = frame[column].map(lambda value: Path(str(value)).name)
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, algorithm, instance), group in frame.groupby(["Benchmark", "Algorithm", "Instance"], sort=True):
        bks = float(group["BKS"].iloc[0])
        best = float(group["Reward"].max())
        mean = float(group["Reward"].mean())
        rows.append({
            "Benchmark": benchmark,
            "Algorithm": algorithm,
            "Instance": instance,
            "BKS": bks,
            "Best_Reward": best,
            "Mean_Reward": mean,
            "Best_Gap_Percent": gap(bks, best),
            "Mean_Gap_Percent": gap(bks, mean),
            "Mean_Runtime_Seconds": float(group["Runtime_Seconds"].mean()),
            "Mean_Completed_Iterations": float(group["Completed_Iterations"].mean()),
        })
    return pd.DataFrame(rows)


def qa_summary(fixed: pd.DataFrame, symmetric: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    blocks = [("Fixed iteration", fixed), ("Equal time", pd.concat([symmetric, runtime], ignore_index=True))]
    rows = []
    for protocol, block in blocks:
        agg = aggregate(block)
        for (benchmark, algorithm), group in agg.groupby(["Benchmark", "Algorithm"], sort=True):
            raw = block[(block["Benchmark"] == benchmark) & (block["Algorithm"] == algorithm)]
            rows.append({
                "Protocol": protocol,
                "Benchmark": benchmark,
                "Algorithm": algorithm,
                "Instances": int(group["Instance"].nunique()),
                "Runs": int(len(raw)),
                "Feasible_Runs": int(raw["Feasible"].sum()),
                "Best_Seed_Gap_Percent": float(group["Best_Gap_Percent"].mean()),
                "Mean_Seed_Gap_Percent": float(group["Mean_Gap_Percent"].mean()),
                "BKS_Hits": int((group["Best_Gap_Percent"] <= 1e-12).sum()),
                "Wall_Clock_Runtime_Per_Seed": float(group["Mean_Runtime_Seconds"].mean()),
                "Mean_Completed_Iterations": float(group["Mean_Completed_Iterations"].mean()),
                "Within_Budget_Runs": None if "Within_Budget" not in raw else int(raw["Within_Budget"].sum()),
            })
    return pd.DataFrame(rows)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(values) - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def paired_stats(reference: pd.Series, comparator: pd.Series) -> dict:
    diff = reference.to_numpy(float) - comparator.to_numpy(float)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return {"W": 0.0, "p": 1.0, "Rank_Biserial": 0.0, "Wins": 0, "Ties": len(diff), "Losses": 0}
    result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
    ranks = pd.Series(np.abs(nz)).rank(method="average").to_numpy()
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    return {
        "W": float(result.statistic), "p": float(result.pvalue),
        "Rank_Biserial": float((pos - neg) / (pos + neg)),
        "Wins": int((diff > 0).sum()), "Ties": int((diff == 0).sum()), "Losses": int((diff < 0).sum()),
    }


def reported_statistics(fixed: pd.DataFrame, symmetric: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    specs = [
        ("Fixed iteration", aggregate(fixed), ["Ejection OFF", "GRASP", "ILS", "VNS"]),
        ("Equal time", aggregate(pd.concat([symmetric, runtime], ignore_index=True)), ["Ejection OFF", "GRASP", "ILS", "VNS", "PyVRP"]),
    ]
    for protocol, agg, comparators in specs:
        for benchmark, data in agg.groupby("Benchmark"):
            local = []
            for outcome, column in (("Best seed", "Best_Reward"), ("Mean seed", "Mean_Reward")):
                pvals = []
                pending = []
                for comparator in comparators:
                    pivot = data[data["Algorithm"].isin(["Ejection ON", comparator])].pivot(
                        index="Instance", columns="Algorithm", values=column
                    )
                    stats = paired_stats(pivot["Ejection ON"], pivot[comparator])
                    pvals.append(stats["p"])
                    pending.append((comparator, stats))
                for (comparator, stats), adjusted in zip(pending, holm(pvals)):
                    local.append({
                        "Protocol": protocol, "Benchmark": benchmark, "Outcome": outcome,
                        "Reference": "Ejection ON", "Comparator": comparator,
                        **stats, "Holm_Adjusted_P_Value": adjusted,
                    })
            outputs.extend(local)
    return pd.DataFrame(outputs)


def component_statistics(sensitivity: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    roles = {
        "Ranking": ("density_p3_c2_l1", "random_p3_c2_l1"),
        "Bounded 2-opt": ("density_p3_c2_l1", "density_p3_c2_l0"),
        "P=1": ("density_p3_c2_l1", "density_p1_c2_l1"),
        "P=5": ("density_p3_c2_l1", "density_p5_c2_l1"),
        "C=1": ("density_p3_c2_l1", "density_p3_c1_l1"),
        "C=3": ("density_p3_c2_l1", "density_p3_c3_l1"),
    }
    rows = []
    for benchmark, data in mechanism.groupby("Benchmark"):
        pending = []
        pvals = []
        for label, (reference, comparator) in roles.items():
            sub = data[data["config"].isin([reference, comparator])]
            block = sub.groupby(["instance", "config"])["best_reward"].mean().unstack()
            stats = paired_stats(block[reference], block[comparator])
            pvals.append(stats["p"])
            pending.append((label, reference, comparator, sub, stats))
        for item, adjusted in zip(pending, holm(pvals)):
            label, reference, comparator, sub, stats = item
            rows.append({
                "Benchmark": benchmark, "Component_Contrast": label,
                "Reference_Config": reference, "Comparator_Config": comparator,
                "Mean_Gap_Reference": float(sub[sub["config"] == reference]["gap_percent"].mean()),
                "Mean_Gap_Comparator": float(sub[sub["config"] == comparator]["gap_percent"].mean()),
                **stats, "Holm_Adjusted_P_Value": adjusted,
            })
    return pd.DataFrame(rows)


def benchmark_manifest(
    bks_by_benchmark: dict[str, dict[str, float]],
    chao_paths: dict[str, Path],
    benchmark_root: Path,
) -> pd.DataFrame:
    doi = {"Dang": "10.1016/j.ejor.2013.02.049", "Chao": "10.1016/0377-2217(94)00289-4"}
    rows = []
    dang_root = benchmark_root / "Dang et al., (2013)"
    dang_files = {path.stem: path for path in dang_root.rglob("*.txt")}
    for benchmark, bks_map in bks_by_benchmark.items():
        for instance, bks in sorted(bks_map.items()):
            path = dang_files[instance] if benchmark == "Dang" else chao_paths[instance]
            raw = path.read_bytes()
            set_name = "Dang benchmark" if benchmark == "Dang" else path.parent.name
            relative = f"{path.parent.name}/{path.name}" if benchmark == "Chao" else path.name
            rows.append({
                "Benchmark": benchmark, "Set": set_name, "Instance": instance, "BKS": float(bks),
                "Bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest(),
                "Source_DOI": doi[benchmark], "Expected_Relative_Path": relative,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    fixed_parts = []
    symmetric_parts = []
    runtime_parts = []
    sensitivity_parts = []
    mechanism_parts = []
    bks_maps: dict[str, dict[str, float]] = {}
    chao_paths: dict[str, Path] = {}

    for name, root in (("Dang", args.dang_root), ("Chao", args.chao_root)):
        fixed_on_off, bks_map, paths = normalize_fixed_on_off(name, root)
        fixed_parts.extend([fixed_on_off, normalize_fixed_baselines(name, root, bks_map)])
        symmetric_parts.append(normalize_symmetric(name, root))
        runtime_parts.append(normalize_runtime_baselines(name, root, bks_map))
        sensitivity_parts.append(normalize_focused(name, root, "01_parameter_sensitivity", "raw_runs.csv"))
        mech_stage = "07_mechanism_components" if name == "Dang" else "06_mechanism_components"
        mechanism_parts.append(normalize_focused(name, root, mech_stage, "mechanism_runs.csv"))
        bks_maps[name] = bks_map
        if name == "Chao":
            chao_paths.update(paths)

    fixed = pd.concat(fixed_parts, ignore_index=True).sort_values(["Benchmark", "Algorithm", "Instance", "Seed"])
    symmetric = pd.concat(symmetric_parts, ignore_index=True).sort_values(["Benchmark", "Algorithm", "Instance", "Seed"])
    runtime = pd.concat(runtime_parts, ignore_index=True).sort_values(["Benchmark", "Algorithm", "Instance", "Seed"])
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True).sort_values(["Benchmark", "config", "instance", "seed"])
    mechanism = pd.concat(mechanism_parts, ignore_index=True).sort_values(["Benchmark", "config", "instance", "seed"])
    qa = qa_summary(fixed, symmetric, runtime)
    stats = reported_statistics(fixed, symmetric, runtime)
    component = component_statistics(sensitivity, mechanism)
    manifest = benchmark_manifest(bks_maps, chao_paths, args.benchmark_root)

    budgets = symmetric.groupby(["Benchmark", "Instance"], as_index=False)["Budget_Seconds"].first()
    bks_rows = [
        {"Benchmark": benchmark, "Instance": instance, "BKS": bks}
        for benchmark, mapping in bks_maps.items() for instance, bks in sorted(mapping.items())
    ]
    payload = {
        "metadata": {
            "title": "Detailed unified workers=2 computational results",
            "description": "Corrected individual algorithm-instance-seed results for the Dang and Chao canonical TOP benchmarks.",
            "gap_definition": "100 * (BKS - reward) / BKS",
            "runtime_definition": "Wall-clock runtime per seed in seconds.",
            "batch_workers": 2,
            "worker_definition": "At most two independent runs were active concurrently; each run used one solver process.",
            "benchmarks": {"Dang": 82, "Chao": 157},
            "fixed_rows": len(fixed), "symmetric_rows": len(symmetric),
            "runtime_baseline_rows": len(runtime), "sensitivity_rows": len(sensitivity),
            "mechanism_rows": len(mechanism),
        },
        "benchmark_bks": bks_rows,
        "fixed_iteration": records(fixed),
        "symmetric_wall_clock": records(symmetric),
        "runtime_budget_baselines": records(runtime),
        "runtime_budgets": records(budgets),
        "sensitivity": records(sensitivity),
        "focused_mechanism": records(mechanism),
        "qa_summary": records(qa),
        "reported_statistics": records(stats),
        "component_statistics": records(component),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.to_csv(args.output.parents[1] / "benchmark_manifest.csv", index=False)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
