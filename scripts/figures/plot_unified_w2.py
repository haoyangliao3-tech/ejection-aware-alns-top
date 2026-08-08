from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "figure_source"
SUMMARY = json.loads((DATA / "unified_w2_summary.json").read_text(encoding="utf-8"))
OUT = BASE / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
})

COLORS = {"ON": "#3E7CB1", "OFF": "#A7A9AC", "GRASP": "#68A691", "ILS": "#D4A259", "VNS": "#B07AA1", "PyVRP": "#D06B64"}


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")


datasets = ["Dang", "Chao"]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), constrained_layout=True)

for col, protocol in enumerate(["fixed", "time"]):
    ax = axes[0, col]
    x = np.arange(len(datasets))
    width = 0.34
    on = [SUMMARY[d][protocol]["on"]["best_gap"] for d in datasets]
    off = [SUMMARY[d][protocol]["off"]["best_gap"] for d in datasets]
    ax.bar(x - width / 2, on, width, label="Ejection ON", color=COLORS["ON"])
    ax.bar(x + width / 2, off, width, label="Ejection OFF", color=COLORS["OFF"])
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Average best-seed BKS gap (%)")
    ax.set_title("Fixed 2,500 iterations" if protocol == "fixed" else "Symmetric wall-clock")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

for col, protocol in enumerate(["fixed", "time"]):
    ax = axes[1, col]
    methods = ["on", "grasp", "ils", "vns"] + (["pyvrp"] if protocol == "time" else [])
    labels = ["ON", "GRASP", "ILS", "VNS"] + (["PyVRP"] if protocol == "time" else [])
    x = np.arange(len(methods))
    width = 0.36
    for j, d in enumerate(datasets):
        vals = [SUMMARY[d][protocol][m]["best_gap"] for m in methods]
        ax.bar(x + (j - 0.5) * width, vals, width, label=d, color="#4878A8" if d == "Dang" else "#E08B42", alpha=0.9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Average best-seed BKS gap (%)")
    ax.set_title("Fixed-iteration comparators" if protocol == "fixed" else "Equal-time comparators")
    ax.legend(ncol=2, loc="upper left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

for label, ax in zip("abcd", axes.flat):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")

save(fig, "figure_unified_w2_results")


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
configs = ["off", "k6_l1", "k20_l1", "k100_l1", "k100_l3", "k100_l5", "k100_linf"]
labels = ["OFF", "$K=6$", "$K=20$", "$K=100$", "$L=3$", "$L=5$", "$L=\\infty$"]
for ax, d in zip(axes, datasets):
    lookup = {r["config"]: r for r in SUMMARY[d]["sensitivity"]}
    gaps = [lookup[c]["mean_gap"] for c in configs]
    runtimes = [lookup[c]["mean_runtime"] for c in configs]
    x = np.arange(len(configs))
    bars = ax.bar(x, gaps, color=[COLORS["OFF"]] + ["#88B5D8"] * 3 + ["#78A98A"] * 3)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Mean BKS gap (%)")
    ax.set_title(f"{d} sensitivity subset")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    ax2 = ax.twinx()
    ax2.plot(x, runtimes, color="#C7664B", marker="o", linewidth=1.1, markersize=3)
    ax2.set_ylabel("Runtime per seed (s)", color="#9A4A38")
    ax2.tick_params(axis="y", colors="#9A4A38")
for label, ax in zip("ab", axes):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")
save(fig, "figure_unified_w2_sensitivity")
