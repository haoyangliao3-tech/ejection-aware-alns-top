from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
GEN = BASE / "data" / "figure_source"
OUT = BASE / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = json.loads((GEN / "unified_w2_summary.json").read_text(encoding="utf-8"))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.75,
    "legend.frameon": False,
})

BLUE = "#3E7CB1"
ORANGE = "#D9824B"
GREY = "#A7A9AC"
GRID = "#E6E6E6"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")


def paired_frame(dataset: str, protocol: str) -> pd.DataFrame:
    path = GEN / f"{dataset.lower()}_{protocol}_instance_summary.csv"
    data = pd.read_csv(path)
    return data[data["algorithm"].isin(["on", "off"])].pivot(
        index="instance", columns="algorithm", values="best_gap"
    ).dropna()


# Figure contract: show that the paired advantage is distributed across instances
# in both benchmark families and under both stopping rules.
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
panels = [
    ("Dang", "fixed", "Dang: fixed 2,500 iterations"),
    ("Chao", "fixed", "Chao: fixed 2,500 iterations"),
    ("Dang", "time", "Dang: symmetric wall-clock"),
    ("Chao", "time", "Chao: symmetric wall-clock"),
]
for ax, (dataset, protocol, title) in zip(axes.flat, panels):
    pair = paired_frame(dataset, protocol)
    color = BLUE if dataset == "Dang" else ORANGE
    maximum = max(float(pair.max().max()), 0.5)
    pad = maximum * 0.05
    ax.plot([0, maximum + pad], [0, maximum + pad], color=GREY, lw=1.0, ls="--", zorder=1)
    ax.scatter(pair["off"], pair["on"], s=18, color=color, alpha=0.72,
               edgecolor="white", linewidth=0.35, zorder=2)
    ax.set_xlim(-pad * 0.15, maximum + pad)
    ax.set_ylim(-pad * 0.15, maximum + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ejection OFF best-seed BKS gap (%)")
    ax.set_ylabel("Ejection ON best-seed BKS gap (%)")
    ax.set_title(title)
    ax.grid(color=GRID, linewidth=0.55)
    wtl = SUMMARY[dataset][f"{protocol}_comparisons"]["off"]["best"]
    ax.text(0.04, 0.96, f"W/T/L = {wtl['wins']}/{wtl['ties']}/{wtl['losses']}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.text(0.96, 0.05, "below diagonal favors ON", transform=ax.transAxes,
            ha="right", va="bottom", color="#555555", fontsize=6.5)
for label, ax in zip("abcd", axes.flat):
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontweight="bold",
            fontsize=8, va="bottom")
save(fig, "figure_ch5_paired_instance_effects")


# Figure contract: distinguish mechanism activity from focused run-level effects.
fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45])
ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[1, :])

datasets = ["Dang", "Chao"]
colors = [BLUE, ORANGE]
x = np.arange(2)
success = [100 * SUMMARY[d]["mechanism"]["success_rate"] for d in datasets]
ax_a.bar(x, success, color=colors, width=0.58)
ax_a.set_xticks(x, datasets)
ax_a.set_ylabel("Successful commits / attempts (%)")
ax_a.set_title("Compound-repair success rate")
ax_a.grid(axis="y", color=GRID, linewidth=0.55)
for i, value in enumerate(success):
    ax_a.text(i, value + max(success) * 0.03, f"{value:.2f}%", ha="center", va="bottom")

metric_labels = ["Bounded 2-opt\nshare", "Module runtime\nshare"]
width = 0.34
for j, dataset in enumerate(datasets):
    values = [
        100 * SUMMARY[dataset]["mechanism"]["two_opt_share"],
        100 * SUMMARY[dataset]["mechanism"]["module_runtime_share"],
    ]
    ax_b.bar(np.arange(2) + (j - 0.5) * width, values, width,
             color=colors[j], label=dataset)
ax_b.set_xticks(np.arange(2), metric_labels)
ax_b.set_ylabel("Share (%)")
ax_b.set_title("Participation and computational share")
ax_b.legend(ncol=2, loc="upper left")
ax_b.grid(axis="y", color=GRID, linewidth=0.55)

component_keys = ["ranking", "two_opt", "positions", "positions_high", "ejections_low", "ejections_high"]
component_labels = [
    "Density vs random",
    "$L=1$ vs $L=0$",
    "$P=3$ vs $P=1$",
    "$P=3$ vs $P=5$",
    "$C=2$ vs $C=1$",
    "$C=2$ vs $C=3$",
]
y = np.arange(len(component_keys))
for j, dataset in enumerate(datasets):
    deltas = [
        SUMMARY[dataset]["components"][key]["mean_gap_comparator"]
        - SUMMARY[dataset]["components"][key]["mean_gap_reference"]
        for key in component_keys
    ]
    ax_c.barh(y + (j - 0.5) * 0.34, deltas, 0.34, color=colors[j], label=dataset)
ax_c.axvline(0, color="#444444", linewidth=0.8)
ax_c.set_yticks(y, component_labels)
ax_c.invert_yaxis()
ax_c.set_xlabel("Comparator gap minus reference gap (percentage points)")
ax_c.set_title("Focused component contrasts (nine instance blocks per benchmark)")
ax_c.legend(ncol=2, loc="lower right")
ax_c.grid(axis="x", color=GRID, linewidth=0.55)
ax_c.text(0.99, 0.98, "positive values favor the reference setting",
          transform=ax_c.transAxes, ha="right", va="top", color="#555555", fontsize=6.5)

for label, ax in zip("abc", [ax_a, ax_b, ax_c]):
    ax.text(-0.13 if ax is not ax_c else -0.06, 1.04, label, transform=ax.transAxes,
            fontweight="bold", fontsize=8, va="bottom")
save(fig, "figure_ch5_mechanism_components")
