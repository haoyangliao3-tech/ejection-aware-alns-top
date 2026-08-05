from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


COLORS = {
    "Small": "#A8C4E0",
    "Medium": "#4D83B5",
    "Large": "#C85C54",
    "hero": "#0F4D92",
    "neutral": "#666666",
    "grid": "#D9D9D9",
}


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = average_ranks(x)
    ry = average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def panel_label(ax, label: str) -> None:
    ax.text(-0.065, 1.035, label, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom")


def read_rows(path: Path) -> list[dict]:
    numeric = {
        "customers", "vehicles", "bks", "best_reward", "mean_reward",
        "best_gap_percent", "mean_gap_percent", "mean_runtime_seconds",
        "min_runtime_seconds", "max_runtime_seconds", "completed_seeds",
    }
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in numeric:
                row[key] = float(row[key])
            rows.append(row)
    if len(rows) != 82:
        raise ValueError(f"Expected 82 instances, found {len(rows)}")
    return rows


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = 100.0 * np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = args.source_csv.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source)

    customers = np.array([row["customers"] for row in rows])
    best_gap = np.array([row["best_gap_percent"] for row in rows])
    mean_gap = np.array([row["mean_gap_percent"] for row in rows])
    runtime = np.array([row["mean_runtime_seconds"] for row in rows])
    groups = np.array([row["size_group"] for row in rows])
    instances = np.array([row["instance"] for row in rows])

    width_in = 183.0 / 25.4
    height_in = 137.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0], hspace=0.52, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # a: hero panel, ranked best-seed gaps.
    order = np.argsort(best_gap, kind="mergesort")
    x_rank = np.arange(1, len(rows) + 1)
    ax_a.plot(x_rank, best_gap[order], color="#B9B9B9", lw=0.8, zorder=1)
    for group in ("Small", "Medium", "Large"):
        mask = groups[order] == group
        ax_a.scatter(x_rank[mask], best_gap[order][mask], s=17, color=COLORS[group],
                     edgecolor="white", linewidth=0.4, zorder=3,
                     label={
                         "Small": r"Small ($\leq$150 customers)",
                         "Medium": "Medium (151-299)",
                         "Large": r"Large ($\geq$300)",
                     }[group])
    for threshold in (1, 3, 5):
        ax_a.axhline(threshold, color="#B7B7B7", lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax_a.text(82.2, threshold, f"{threshold}%", color="#777777", fontsize=6.5,
                  va="center", ha="right")
    bks_hits = int(np.sum(best_gap <= 1e-12))
    within_five = 100.0 * np.mean(best_gap <= 5.0)
    ax_a.text(0.015, 0.93, f"{bks_hits}/82 reached BKS; {within_five:.1f}% within 5%",
              transform=ax_a.transAxes, ha="left", va="top", fontsize=7.1, color="#333333")
    largest = order[-3:][::-1]
    largest_text = "Largest gaps\n" + "\n".join(
        f"{instances[idx].replace('_', '-')} : {best_gap[idx]:.2f}%" for idx in largest
    )
    ax_a.text(0.985, 0.965, largest_text, transform=ax_a.transAxes, ha="right", va="top",
              fontsize=6.4, color="#555555",
              bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D0D0D0", "lw": 0.6})
    ax_a.set_xlim(0, 83)
    ax_a.set_ylim(-0.25, max(10.2, float(best_gap.max()) + 0.8))
    ax_a.set_xticks([1, 20, 40, 60, 82])
    ax_a.set_xlabel("Instances ranked by best-seed BKS gap")
    ax_a.set_ylabel("Best-seed BKS gap (%)")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                fontsize=6.8, handletextpad=0.4, columnspacing=1.4)
    panel_label(ax_a, "a")

    # b: empirical distributions.
    xb, yb = empirical_cdf(best_gap)
    xm, ym = empirical_cdf(mean_gap)
    ax_b.step(xb, yb, where="post", color=COLORS["hero"], lw=1.7, label="Best of 4 seeds")
    ax_b.step(xm, ym, where="post", color=COLORS["neutral"], lw=1.35,
              ls=(0, (4, 2)), label="Mean of 4 seeds")
    ax_b.set_xlim(0, max(12.0, float(mean_gap.max()) + 0.5))
    ax_b.set_ylim(0, 102)
    ax_b.set_xlabel("BKS gap (%)")
    ax_b.set_ylabel("Instances at or below gap (%)")
    ax_b.legend(loc="lower right", fontsize=6.4)
    ax_b.text(0.03, 1.03, f"Median: {np.median(best_gap):.2f}% best; {np.median(mean_gap):.2f}% mean",
              transform=ax_b.transAxes, ha="left", va="bottom", fontsize=6.3, color="#444444")
    panel_label(ax_b, "b")

    # c: size-stratified best gaps.
    group_order = ["Small", "Medium", "Large"]
    values = [best_gap[groups == group] for group in group_order]
    box = ax_c.boxplot(values, widths=0.55, patch_artist=True, showfliers=False,
                       medianprops={"color": "#222222", "lw": 1.0},
                       whiskerprops={"color": "#777777", "lw": 0.8},
                       capprops={"color": "#777777", "lw": 0.8})
    for patch, group in zip(box["boxes"], group_order):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.48)
        patch.set_edgecolor("#666666")
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(20260726)
    for pos, (group, vals) in enumerate(zip(group_order, values), start=1):
        jitter = rng.normal(0.0, 0.055, len(vals))
        ax_c.scatter(pos + jitter, vals, s=13, color=COLORS[group], edgecolor="white",
                     linewidth=0.35, alpha=0.9, zorder=3)
        ax_c.scatter(pos, np.mean(vals), marker="D", s=38, color="#222222",
                     edgecolor="white", linewidth=0.6, zorder=4)
    counts = [int(np.sum(groups == group)) for group in group_order]
    ax_c.set_xticks([1, 2, 3])
    ax_c.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(group_order, counts)])
    ax_c.set_ylabel("Best-seed BKS gap (%)")
    ax_c.set_ylim(-0.25, max(10.2, float(best_gap.max()) + 0.8))
    ax_c.text(0.03, 0.96, "Diamonds: group means", transform=ax_c.transAxes,
              ha="left", va="top", fontsize=6.3, color="#555555")
    panel_label(ax_c, "c")

    # d: runtime scaling using final post-optimization wall-clock records.
    for group in group_order:
        mask = groups == group
        ax_d.scatter(customers[mask], runtime[mask], s=18, color=COLORS[group],
                     edgecolor="white", linewidth=0.4, alpha=0.9)
    positive = runtime > 0
    coeff = np.polyfit(customers[positive], np.log10(runtime[positive]), 1)
    x_line = np.linspace(customers.min(), customers.max(), 120)
    y_line = 10 ** np.polyval(coeff, x_line)
    ax_d.plot(x_line, y_line, color="#555555", lw=1.0, ls=(0, (3, 2)))
    rho = spearman(customers, runtime)
    ax_d.set_yscale("log")
    ax_d.set_xlim(float(customers.min()) - 8, float(customers.max()) + 8)
    ax_d.set_xlabel("Customers")
    ax_d.set_ylabel("Mean wall-clock runtime/seed (s)")
    ax_d.text(0.04, 0.92, rf"Spearman $\rho={rho:.2f}$", transform=ax_d.transAxes,
              ha="left", va="top", fontsize=6.6, color="#444444")
    panel_label(ax_d, "d")

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.tick_params(direction="out", length=2.6, width=0.7, labelsize=6.7)
        ax.yaxis.grid(True, color=COLORS["grid"], lw=0.45, alpha=0.65)
        ax.set_axisbelow(True)

    fig.text(0.5, 0.012,
             r"$K=100$, $L=1$; 2,500 iterations; four seeds per instance (328 feasible runs)",
             ha="center", va="bottom", fontsize=6.7, color="#555555")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.12)

    stem = output / "figure2_benchmark_82_k100_l1_2500"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(source, stem.parent / f"{stem.name}_source_data.csv")
    legend = (
        "Fig. 2 | Distribution and computational profile of Ejection-Aware ALNS "
        "over 82 Dang benchmark instances. a, Ranked best-seed BKS gaps, coloured by instance size; "
        "labels identify the three largest remaining gaps. b, Empirical cumulative distributions of "
        "best-seed and mean-seed BKS gaps over four seeds. c, Best-seed BKS gaps by customer-count "
        "stratum; boxes show the interquartile range and median, points show instances, and diamonds "
        "show group means. d, Mean wall-clock runtime per seed against customer count; the dashed line "
        "is a log-linear visual guide and Spearman rho is descriptive. All runs used K=100, L=1 and "
        "2,500 iterations. All 328 runs were feasible."
    )
    (stem.parent / f"{stem.name}_legend.txt").write_text(legend, encoding="utf-8")
    print(f"rho={rho:.6f}; bks_hits={bks_hits}; within_5={within_five:.6f}")
    for suffix in ("svg", "pdf", "tiff", "png", "csv", "txt"):
        candidate = stem.with_suffix(f".{suffix}") if suffix not in ("csv", "txt") else stem.parent / f"{stem.name}_{'source_data' if suffix == 'csv' else 'legend'}.{suffix}"
        print(candidate)


if __name__ == "__main__":
    main()
