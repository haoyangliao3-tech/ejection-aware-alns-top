from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, rankdata, wilcoxon


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.0
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


BLUE = "#155A9C"
MID_BLUE = "#4D83B5"
LIGHT_BLUE = "#A8C4E0"
GREY = "#6E6E6E"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    config_to_k = {"k6_l1": 6, "k20_l1": 20, "k100_l1": 100}
    data = raw.loc[raw["config"].isin(config_to_k)].copy()
    data["K"] = data["config"].map(config_to_k).astype(int)
    data = data.loc[:, ["instance", "seed", "K", "gap_percent", "runtime_seconds",
                      "iterations", "feasible"]]

    expected_counts = {6: 18, 20: 18, 100: 18}
    if data.groupby("K").size().to_dict() != expected_counts:
        raise ValueError("Expected 18 runs for each of K=6, K=20, and K=100")
    if not data["feasible"].astype(bool).all() or not (data["iterations"] == 5000).all():
        raise ValueError("All selected runs must be feasible 5,000-iteration runs")

    run_gap = data.pivot(index=["instance", "seed"], columns="K", values="gap_percent")
    runtime = data.pivot(index=["instance", "seed"], columns="K", values="runtime_seconds")
    if run_gap.shape != (18, 3) or runtime.shape != (18, 3) or run_gap.isna().any().any():
        raise ValueError("The K configurations are not paired by instance and seed")
    gap = data.pivot_table(
        index="instance", columns="K", values="gap_percent", aggfunc="mean"
    )
    if gap.shape != (9, 3) or gap.isna().any().any():
        raise ValueError("Expected nine complete instance blocks")

    expected = {
        6: (4.892741, 237.560195),
        20: (4.458099, 395.431403),
        100: (3.480757, 874.882191),
    }
    for k, (target_gap, target_runtime) in expected.items():
        if abs(run_gap[k].mean() - target_gap) > 1e-6:
            raise ValueError(f"Unexpected mean gap for K={k}: {run_gap[k].mean()}")
        if abs(runtime[k].mean() - target_runtime) > 1e-6:
            raise ValueError(f"Unexpected mean runtime for K={k}: {runtime[k].mean()}")
    return data, gap


def paired_stat(gap: pd.DataFrame, comparator: int) -> dict[str, float | int | str]:
    difference = (gap[comparator] - gap[100]).to_numpy()
    permutation = PermutationMethod(n_resamples=np.inf, random_state=20260730)
    result = wilcoxon(difference, zero_method="pratt", correction=False,
                      alternative="two-sided", method=permutation)
    ranks = rankdata(np.abs(difference), method="average")
    positive = float(ranks[difference > 0].sum())
    negative = float(ranks[difference < 0].sum())
    rank_biserial = (positive - negative) / (positive + negative)
    tolerance = 1e-12
    return {
        "comparison": f"K=100 vs K={comparator}",
        "comparator": comparator,
        "n_pairs": len(difference),
        "mean_gap_reduction_pp": float(difference.mean()),
        "wins": int((difference > tolerance).sum()),
        "ties": int((np.abs(difference) <= tolerance).sum()),
        "losses": int((difference < -tolerance).sum()),
        "wilcoxon_W": float(result.statistic),
        "raw_p_value": float(result.pvalue),
        "rank_biserial": float(rank_biserial),
    }


def holm_adjust_two(rows: list[dict[str, float | int | str]]) -> None:
    order = np.argsort([float(row["raw_p_value"]) for row in rows])
    adjusted = np.empty(2, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (2 - rank) * float(rows[index]["raw_p_value"])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = float(value)


def build_outputs(data: pd.DataFrame, gap: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = [paired_stat(gap, 6), paired_stat(gap, 20)]
    holm_adjust_two(stats)

    means = data.groupby("K", as_index=False).agg(
        mean_gap_percent=("gap_percent", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    ).sort_values("K")
    transitions = []
    for start, end in [(6, 20), (20, 100)]:
        first = means.loc[means["K"] == start].iloc[0]
        second = means.loc[means["K"] == end].iloc[0]
        gap_reduction = float(first["mean_gap_percent"] - second["mean_gap_percent"])
        added_runtime = float(second["mean_runtime_seconds"] - first["mean_runtime_seconds"])
        transitions.append({
            "comparison": f"K={start} to K={end}",
            "gap_reduction_pp": gap_reduction,
            "added_runtime_seconds": added_runtime,
            "gap_reduction_pp_per_100_added_seconds": 100.0 * gap_reduction / added_runtime,
        })

    source = data.groupby(["instance", "K"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        mean_gap_percent=("gap_percent", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    ).sort_values(["instance", "K"])
    reference = gap[100].rename("k100_gap_percent")
    source = source.merge(reference, left_on="instance", right_index=True,
                          validate="many_to_one")
    source["paired_gap_difference_vs_k100_pp"] = source["mean_gap_percent"] - source["k100_gap_percent"]
    source.to_csv(output_dir / "figure_k_marginal_analysis_source_data.csv", index=False)
    pd.DataFrame(stats).to_csv(output_dir / "figure_k_marginal_analysis_statistics.csv", index=False)
    pd.DataFrame(transitions).to_csv(output_dir / "figure_k_marginal_analysis_marginal_returns.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 72 / 25.4),
                             gridspec_kw={"width_ratios": [0.92, 1.08]})
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.88, wspace=0.38)

    # Panel a: paired run-level gap differences.
    ax = axes[0]
    differences = [(gap[6] - gap[100]).to_numpy(), (gap[20] - gap[100]).to_numpy()]
    box = ax.boxplot(differences, positions=[1, 2], widths=0.46, patch_artist=True,
                     showfliers=False, medianprops={"color": BLUE, "linewidth": 1.2},
                     whiskerprops={"color": GREY, "linewidth": 0.8},
                     capprops={"color": GREY, "linewidth": 0.8})
    for patch, color in zip(box["boxes"], [LIGHT_BLUE, "#D5E2EF"]):
        patch.set_facecolor(color)
        patch.set_edgecolor(MID_BLUE)
        patch.set_linewidth(0.9)
    rng = np.random.default_rng(20260729)
    for position, values in zip([1, 2], differences):
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(position + jitter, values, s=12, color=GREY, alpha=0.70,
                   edgecolor="white", linewidth=0.25, zorder=3)
        ax.scatter(position, np.mean(values), marker="D", s=27, color=BLUE,
                   edgecolor="white", linewidth=0.45, zorder=4)
    ax.axhline(0, color="#8C8C8C", lw=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["$K=6$ vs\n$K=100$", "$K=20$ vs\n$K=100$"])
    ax.set_ylabel("Paired BKS-gap difference\n(comparator $-$ $K=100$; percentage points)")
    ax.set_ylim(-3.2, 7.6)
    ax.text(1, 7.15, "$p_{\\mathrm{Holm}}=0.0234$", ha="center", va="top", fontsize=6.8)
    ax.text(2, 7.15, "$p_{\\mathrm{Holm}}=0.0625$", ha="center", va="top", fontsize=6.8)
    ax.text(0.02, 0.02, "Positive values favor $K=100$", transform=ax.transAxes,
            color=GREY, fontsize=6.4, va="bottom")
    panel_label(ax, "a")

    # Panel b: mean quality-runtime path and marginal returns.
    ax = axes[1]
    x = means["mean_runtime_seconds"].to_numpy()
    y = means["mean_gap_percent"].to_numpy()
    ks = means["K"].astype(int).to_numpy()
    ax.plot(x, y, color=MID_BLUE, lw=1.25, zorder=1)
    ax.scatter(x, y, s=[34, 38, 46], color=[LIGHT_BLUE, MID_BLUE, BLUE],
               edgecolor="white", linewidth=0.55, zorder=3)
    offsets = {6: (-5, 7), 20: (-3, -15), 100: (-2, -13)}
    for xi, yi, k in zip(x, y, ks):
        dx, dy = offsets[int(k)]
        ax.annotate(f"$K={k}$", (xi, yi), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.8, fontweight="bold", color=BLUE)
    first, second = transitions
    ax.annotate("$-0.435$ pp; $+157.9$ s\n0.275 pp per 100 s",
                xy=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2), xytext=(12, 0),
                textcoords="offset points", fontsize=6.2, color=GREY, va="center")
    ax.annotate("$-0.977$ pp; $+479.5$ s\n0.204 pp per 100 s",
                xy=((x[1] + x[2]) / 2, (y[1] + y[2]) / 2), xytext=(7, 5),
                textcoords="offset points", fontsize=6.2, color=GREY, va="center")
    ax.set_xlabel("Mean wall-clock runtime per seed (s)")
    ax.set_ylabel("Mean BKS gap (%)")
    ax.set_xlim(175, 940)
    ax.set_ylim(3.20, 5.12)
    ax.grid(axis="both", color="#E5E5E5", lw=0.55)
    ax.text(0.98, 0.98, "Lower gap is better", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.4, color=GREY)
    panel_label(ax, "b")

    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 300},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(output_dir / f"figure_k_marginal_analysis{suffix}",
                    bbox_inches="tight", **kwargs)
    plt.close(fig)

    legend = (
        "Marginal analysis of the blocked-customer budget K at L=1. "
        "a, Paired BKS-gap differences for nine instance means over two seeds; "
        "positive values favor K=100. Boxes show the interquartile range and median, "
        "points show paired observations, and diamonds show means. P-values are from "
        "two-sided exact-permutation Wilcoxon signed-rank tests with Pratt zero handling and Holm "
        "adjustment across the two planned contrasts. b, Mean quality--runtime path "
        "for K=6, 20, and 100, with incremental gap reductions, additional runtime, "
        "and marginal returns. All configurations used nine instances, two seeds, "
        "5,000 iterations, and L=1."
    )
    (output_dir / "figure_k_marginal_analysis_legend.txt").write_text(legend, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_runs_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    data, gap = load_data(args.raw_runs_csv)
    build_outputs(data, gap, args.output_dir)


if __name__ == "__main__":
    main()
