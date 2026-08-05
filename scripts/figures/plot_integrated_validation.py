from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    "hero_light": "#A8C4E0",
    "neutral": "#767676",
    "neutral_light": "#D8D8D8",
    "paper": "#9A4D8E",
    "positive": "#2E8B57",
}


def panel_label(ax, label: str) -> None:
    ax.text(-0.11, 1.03, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def load_and_validate(ablation_path: Path, profile_path: Path,
                      fixed_path: Path, time_path: Path):
    ablation = pd.read_csv(ablation_path)
    profile = pd.read_csv(profile_path)[["instance", "customers", "vehicles", "size_group"]]
    paired = ablation.merge(profile, on="instance", how="left", validate="one_to_one")
    fixed = pd.read_csv(fixed_path)
    time = pd.read_csv(time_path)
    if len(paired) != 82 or paired["size_group"].isna().any():
        raise ValueError("The paired ablation source must contain 82 classified instances")
    if paired["size_group"].value_counts().to_dict() != {"Medium": 47, "Small": 20, "Large": 15}:
        raise ValueError("Unexpected instance-size counts")
    best_delta = paired["gap_best_off"] - paired["gap_best_on"]
    wins = int((best_delta > 1e-12).sum())
    ties = int((best_delta.abs() <= 1e-12).sum())
    losses = int((best_delta < -1e-12).sum())
    if (wins, ties, losses) != (60, 13, 9):
        raise ValueError(f"Unexpected best-seed W/T/L: {(wins, ties, losses)}")
    checks = {
        "best_off": paired["gap_best_off"].mean(),
        "best_on": paired["gap_best_on"].mean(),
        "mean_off": paired["gap_mean_off"].mean(),
        "mean_on": paired["gap_mean_on"].mean(),
    }
    expected = {"best_off": 3.707, "best_on": 1.820, "mean_off": 4.998, "mean_on": 3.038}
    for key, target in expected.items():
        if abs(checks[key] - target) > 0.0015:
            raise ValueError(f"Aggregate check failed for {key}: {checks[key]}")
    return paired, fixed, time, (wins, ties, losses)


def plot_protocol(ax, summary: pd.DataFrame, methods: list[str], labels: list[str], title: str):
    indexed = summary.set_index("algorithm").loc[methods]
    y = np.arange(len(methods))[::-1]
    for yi, method, label in zip(y, methods, labels):
        row = indexed.loc[method]
        reproduced = {"kim_alns"}
        color = COLORS["hero"] if method == "proposed" else (COLORS["paper"] if method in reproduced else COLORS["neutral"])
        ax.plot([row["mean_best_gap"], row["mean_gap"]], [yi, yi], color=color,
                lw=1.35, alpha=0.78, zorder=1)
        ax.scatter(row["mean_best_gap"], yi, s=28, color=color, edgecolor="white",
                   linewidth=0.45, zorder=3)
        ax.scatter(row["mean_gap"], yi, s=30, facecolor="white", edgecolor=color,
                   linewidth=1.15, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    upper = max(19.0, float(indexed[["mean_best_gap", "mean_gap"]].max().max()) + 1.0)
    upper = 5.0 * np.ceil(upper / 5.0)
    ax.set_xlim(0, upper)
    ax.set_xticks(np.arange(0, upper, 5.0))
    ax.set_xlabel("BKS gap (%)")
    ax.set_title(title, fontsize=7.4, pad=4, fontweight="bold")
    ax.axvline(0, color="#AAAAAA", lw=0.6)
    ax.tick_params(axis="y", length=0)
    if len(methods) > 6:
        ax.tick_params(axis="y", labelsize=6.2)


def build_figure(paired: pd.DataFrame, fixed: pd.DataFrame, time: pd.DataFrame,
                 wtl: tuple[int, int, int], output_dir: Path) -> None:
    width_in = 183.0 / 25.4
    height_in = 135.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.24, 1.0, 1.0],
                          height_ratios=[1.04, 1.0], hspace=0.42, wspace=0.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # a: paired ablation hero panel.
    limit = 21.0
    x = paired["gap_best_off"].to_numpy()
    y = paired["gap_best_on"].to_numpy()
    xx = np.linspace(0, limit, 250)
    ax_a.fill_between(xx, 0, xx, color="#EDF4FA", zorder=0)
    ax_a.plot(xx, xx, color="#7A7A7A", lw=0.9, ls=(0, (4, 2)), zorder=1)
    for group in ("Small", "Medium", "Large"):
        subset = paired[paired["size_group"] == group]
        ax_a.scatter(subset["gap_best_off"], subset["gap_best_on"], s=24,
                     color=COLORS[group], edgecolor="white", linewidth=0.45,
                     alpha=0.92, label={
                         "Small": r"Small ($n=20$)",
                         "Medium": r"Medium ($n=47$)",
                         "Large": r"Large ($n=15$)",
                     }[group], zorder=3)
    ax_a.set_xlim(-0.3, limit)
    ax_a.set_ylim(-0.3, limit)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Ejection OFF best-seed BKS gap (%)")
    ax_a.set_ylabel("Ejection ON best-seed BKS gap (%)")
    fig.text(0.055, 0.875, f"W/T/L = {wtl[0]}/{wtl[1]}/{wtl[2]}",
             ha="left", va="top", fontsize=6.7, fontweight="bold",
             color=COLORS["hero"])
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["Small"],
               markeredgecolor="white", markersize=5, label=r"Small ($n=20$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["Medium"],
               markeredgecolor="white", markersize=5, label=r"Medium ($n=47$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["Large"],
               markeredgecolor="white", markersize=5, label=r"Large ($n=15$)"),
    ], loc="upper left", bbox_to_anchor=(0.055, 0.835), ncol=1,
       fontsize=6.3, handletextpad=0.35, labelspacing=0.25)
    panel_label(ax_a, "a")

    inset = inset_axes(ax_a, width="43%", height="36%", loc="upper right", borderpad=0.9)
    inset.plot([0, 6], [0, 6], color="#8A8A8A", lw=0.7, ls=(0, (3, 2)))
    for group in ("Small", "Medium", "Large"):
        subset = paired[paired["size_group"] == group]
        inset.scatter(subset["gap_best_off"], subset["gap_best_on"], s=9,
                      color=COLORS[group], edgecolor="white", linewidth=0.25, alpha=0.88)
    inset.set_xlim(-0.1, 6)
    inset.set_ylim(-0.1, 6)
    inset.set_xticks([0, 3, 6])
    inset.set_yticks([0, 3, 6])
    inset.tick_params(labelsize=5.5, length=2)
    inset.spines["top"].set_visible(True)
    inset.spines["right"].set_visible(True)
    inset.set_title("0-6% detail", fontsize=5.8, pad=2)

    # b: size-stratified reductions.
    groups = ["Small", "Medium", "Large"]
    rng = np.random.default_rng(20260726)
    best_values = [(paired.loc[paired["size_group"] == g, "gap_best_off"] -
                    paired.loc[paired["size_group"] == g, "gap_best_on"]).to_numpy() for g in groups]
    mean_values = [(paired.loc[paired["size_group"] == g, "gap_mean_off"] -
                    paired.loc[paired["size_group"] == g, "gap_mean_on"]).to_numpy() for g in groups]
    positions_best = np.arange(1, 4) - 0.17
    positions_mean = np.arange(1, 4) + 0.17
    for positions, values, face, edge in (
        (positions_best, best_values, COLORS["hero"], COLORS["hero"]),
        (positions_mean, mean_values, "white", COLORS["hero_light"]),
    ):
        box = ax_b.boxplot(values, positions=positions, widths=0.25, patch_artist=True,
                           showfliers=False,
                           medianprops={"color": "#222222", "lw": 0.9},
                           whiskerprops={"color": edge, "lw": 0.75},
                           capprops={"color": edge, "lw": 0.75})
        for patch in box["boxes"]:
            patch.set_facecolor(face)
            patch.set_edgecolor(edge)
            patch.set_alpha(0.65 if face != "white" else 1.0)
            patch.set_linewidth(0.8)
        for pos, vals in zip(positions, values):
            jitter = rng.normal(0, 0.025, len(vals))
            ax_b.scatter(pos + jitter, vals, s=8, facecolor=face,
                         edgecolor=edge if face == "white" else "white",
                         linewidth=0.3, alpha=0.7, zorder=3)
    ax_b.axhline(0, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax_b.set_xticks([1, 2, 3])
    ax_b.set_xticklabels(["Small\n$n=20$", "Medium\n$n=47$", "Large\n$n=15$"])
    ax_b.set_ylabel("BKS gap reduction\n(OFF - ON; percentage points)")
    ax_b.text(0.99, 0.95, "Positive favors Ejection ON", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=6.2, color=COLORS["positive"])
    ax_b.legend(handles=[
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["hero"],
               markeredgecolor=COLORS["hero"], markersize=5, label="Best seed"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["hero_light"], markersize=5, label="Mean seed"),
    ], loc="upper left", ncol=2, fontsize=6.3, handletextpad=0.35, columnspacing=0.9)
    panel_label(ax_b, "b")

    # c-d: complete-method validation under distinct protocols.
    plot_protocol(ax_c, fixed,
                  ["proposed", "grasp", "ils", "vns"],
                  ["Ejection ON", "GRASP", "ILS", "VNS"],
                  "Fixed iteration")
    plot_protocol(ax_d, time,
                  ["proposed", "kim_alns", "grasp", "ils", "vns", "pyvrp"],
                  ["Ejection ON ref.", "Kim ALNS", "GRASP", "ILS", "VNS", "PyVRP"],
                  "Runtime budget")
    panel_label(ax_c, "c")
    panel_label(ax_d, "d")
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555",
               markersize=4.5, lw=0, label="Best-seed gap"),
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="white",
               markeredgewidth=1.0, markersize=4.5, lw=0, label="Mean-seed gap"),
    ], loc="lower center", bbox_to_anchor=(0.755, 0.005), ncol=2,
       fontsize=6.4, handletextpad=0.35, columnspacing=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "figure4_integrated_validation"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    paired_source = paired.copy()
    paired_source.insert(0, "panel", "paired_ablation")
    fixed_source = fixed.copy()
    fixed_source.insert(0, "panel", "fixed_iteration")
    time_source = time.copy()
    time_source.insert(0, "panel", "runtime_budget")
    source = pd.concat([paired_source, fixed_source, time_source], ignore_index=True, sort=False)
    source.to_csv(output_dir / "figure4_integrated_validation_source_data.csv", index=False)
    (output_dir / "figure4_integrated_validation_legend.txt").write_text(
        "Figure 5 | Controlled validation of Ejection-Aware Repair and the complete method. "
        "a, Paired best-seed BKS gaps for Ejection OFF and Ejection ON across 82 instances; "
        "points below the identity line favor Ejection ON and the inset enlarges the 0-6% region. "
        "b, Instance-level best-seed and mean-seed BKS gap reductions stratified by customer-count group; "
        "boxes show the interquartile range and median and points show instances. "
        "c, Fixed-iteration comparison of Ejection ON with the solver-free baselines. "
        "d, Time-stopped comparator runs under instance-specific budgets, shown against the existing 2,500-iteration "
        "Ejection ON fixed-iteration reference. "
        "All comparisons use 82 instances and four seeds; lower BKS gaps are better.\n",
        encoding="utf-8",
    )
    print(f"saved={base} wtl={wtl}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_csv", type=Path)
    parser.add_argument("profile_csv", type=Path)
    parser.add_argument("fixed_summary_csv", type=Path)
    parser.add_argument("time_summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paired, fixed, time, wtl = load_and_validate(
        args.ablation_csv.resolve(), args.profile_csv.resolve(),
        args.fixed_summary_csv.resolve(), args.time_summary_csv.resolve(),
    )
    build_figure(paired, fixed, time, wtl, args.output_dir.resolve())


if __name__ == "__main__":
    main()
