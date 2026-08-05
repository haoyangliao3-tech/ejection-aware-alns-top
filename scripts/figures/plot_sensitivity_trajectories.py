"""Create the publication figure for the 5,000-iteration sensitivity runs.

The input is the checkpoint-level trajectory CSV produced by the experiment.
No solver code or experimental result is modified by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


CONFIG_STYLE = {
    "off": {
        "label": "Repair OFF",
        "color": "#4D4D4D",
        "linestyle": (0, (4.0, 2.0)),
        "linewidth": 1.35,
        "zorder": 4,
    },
    "k6_l1": {
        "label": r"$K=6$, $L=1$",
        "color": "#9A9A9A",
        "linestyle": (0, (1.0, 1.2)),
        "linewidth": 1.0,
        "zorder": 3,
    },
    "k20_l1": {
        "label": r"$K=20$, $L=1$",
        "color": "#E69F00",
        "linestyle": (0, (5.0, 1.7, 1.0, 1.7)),
        "linewidth": 1.05,
        "zorder": 3,
    },
    "k100_l1": {
        "label": r"$K=100$, $L=1$",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.75,
        "zorder": 6,
    },
    "k100_l3": {
        "label": r"$K=100$, $L=3$",
        "color": "#009E73",
        "linestyle": (0, (3.0, 1.2)),
        "linewidth": 1.1,
        "zorder": 3,
    },
    "k100_l5": {
        "label": r"$K=100$, $L=5$",
        "color": "#CC79A7",
        "linestyle": (0, (6.0, 1.5)),
        "linewidth": 1.1,
        "zorder": 3,
    },
    "k100_linf": {
        "label": r"$K=100$, $L=\infty$",
        "color": "#D55E00",
        "linestyle": (0, (2.0, 1.0, 1.0, 1.0)),
        "linewidth": 1.1,
        "zorder": 3,
    },
}


INSTANCE_GRID = [
    ["bier127_gen1_m3", "bier127_gen2_m3", "bier127_gen3_m3"],
    ["pr299_gen1_m3", "pr299_gen2_m3", "pr299_gen3_m3"],
    ["rd400_gen1_m3", "rd400_gen2_m3", "rd400_gen3_m3"],
]

ROW_LABELS = [
    "Small\n126 customers",
    "Medium\n298 customers",
    "Large\n399 customers",
]

PANEL_LABELS = list("abcdefghi")


def configure_matplotlib() -> None:
    """Use editable text and a compact Nature-style typographic system."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def validate_and_summarize(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "size",
        "instance",
        "config",
        "seed",
        "iteration",
        "gap_percent",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    expected_instances = {item for row in INSTANCE_GRID for item in row}
    observed_instances = set(df["instance"].unique())
    if observed_instances != expected_instances:
        raise ValueError(
            "Unexpected instance set. "
            f"Missing={sorted(expected_instances - observed_instances)}, "
            f"extra={sorted(observed_instances - expected_instances)}"
        )

    observed_configs = set(df["config"].unique())
    expected_configs = set(CONFIG_STYLE)
    if observed_configs != expected_configs:
        raise ValueError(
            "Unexpected configuration set. "
            f"Missing={sorted(expected_configs - observed_configs)}, "
            f"extra={sorted(observed_configs - expected_configs)}"
        )

    expected_iterations = np.arange(0, 5001, 50)
    for keys, group in df.groupby(["instance", "config", "seed"], sort=False):
        observed = np.sort(group["iteration"].to_numpy())
        if not np.array_equal(observed, expected_iterations):
            raise ValueError(f"Incomplete 50-iteration checkpoints for run {keys}")

    seeds_per_cell = df.groupby(["instance", "config"])["seed"].nunique()
    if not (seeds_per_cell == 2).all():
        raise ValueError("Every instance/configuration cell must contain exactly two seeds")

    summary = (
        df.groupby(["size", "instance", "config", "iteration"], as_index=False)
        .agg(
            mean_gap_percent=("gap_percent", "mean"),
            min_gap_percent=("gap_percent", "min"),
            max_gap_percent=("gap_percent", "max"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["instance", "config", "iteration"])
    )
    return summary


def panel_ylim(panel: pd.DataFrame) -> tuple[float, float]:
    """Set a truthful local scale while keeping zero visible when appropriate."""
    low = float(panel["min_gap_percent"].min())
    high = float(panel["max_gap_percent"].max())
    span = max(high - low, 1.0)
    bottom = max(0.0, low - 0.08 * span)
    top = high + 0.08 * span
    return bottom, top


def make_figure(summary: pd.DataFrame) -> plt.Figure:
    width_in = 183.0 / 25.4
    height_in = 166.0 / 25.4
    fig, axes = plt.subplots(3, 3, figsize=(width_in, height_in), sharex=True)

    for row_idx, row in enumerate(INSTANCE_GRID):
        for col_idx, instance in enumerate(row):
            ax = axes[row_idx, col_idx]
            panel = summary.loc[summary["instance"] == instance]

            for config, style in CONFIG_STYLE.items():
                curve = panel.loc[panel["config"] == config].sort_values("iteration")
                x = curve["iteration"].to_numpy()
                y = curve["mean_gap_percent"].to_numpy()

                if config in {"off", "k100_l1"}:
                    ax.fill_between(
                        x,
                        curve["min_gap_percent"].to_numpy(),
                        curve["max_gap_percent"].to_numpy(),
                        step="post",
                        color=style["color"],
                        alpha=0.09,
                        linewidth=0,
                        zorder=1,
                    )

                ax.step(
                    x,
                    y,
                    where="post",
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    zorder=style["zorder"],
                    solid_capstyle="butt",
                    dash_capstyle="butt",
                )

            short_name = instance.replace("_m3", "").replace("_gen", " · Gen")
            ax.set_title(short_name, pad=3.0)
            ax.set_xlim(0, 5000)
            ax.set_ylim(*panel_ylim(panel))
            ax.set_xticks([0, 2500, 5000], labels=["0", "2.5k", "5k"])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
            ax.grid(axis="x", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", pad=1.5)
            ax.text(
                0.015,
                0.985,
                PANEL_LABELS[row_idx * 3 + col_idx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                fontweight="bold",
                color="#111111",
            )

    fig.supxlabel("Iterations", y=0.058, fontsize=7.5)
    fig.supylabel("Gap to best-known solution (%)", x=0.055, fontsize=7.5)

    row_y = [0.735, 0.445, 0.155]
    for label, y in zip(ROW_LABELS, row_y):
        fig.text(
            0.012,
            y,
            label,
            ha="left",
            va="center",
            rotation=90,
            fontsize=6.7,
            color="#333333",
            linespacing=0.95,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        for style in CONFIG_STYLE.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.992),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.25,
        handletextpad=0.45,
        borderaxespad=0,
    )

    fig.text(
        0.52,
        0.014,
        "Lines show the mean of two seeds; shading shows the seed range for Repair OFF and $K=100$, $L=1$.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.992,
        bottom=0.105,
        top=0.895,
        wspace=0.29,
        hspace=0.40,
    )
    return fig


def export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "bbox_inches": "tight",
        "pad_inches": 0.025,
        "facecolor": "white",
        "transparent": False,
    }
    fig.savefig(output_dir / f"{stem}.svg", format="svg", **common)
    fig.savefig(
        output_dir / f"{stem}.pdf",
        format="pdf",
        metadata={
            "Title": "Sensitivity trajectories across nine benchmark instances",
            "Subject": "Mean gap-to-BKS trajectories from two seeds",
            "Creator": "Matplotlib",
        },
        **common,
    )
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, **common)
    fig.savefig(
        output_dir / f"{stem}.tiff",
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/sensitivity_history_5000/trajectory.csv"),
        help="Checkpoint-level trajectory CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sensitivity_history_5000/figures"),
        help="Directory for figure files and processed Source Data.",
    )
    parser.add_argument(
        "--stem",
        default="figure_sensitivity_trajectories",
        help="Base filename for exported figure files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    trajectory = pd.read_csv(args.input)
    summary = validate_and_summarize(trajectory)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / f"{args.stem}_source_data.csv"
    summary.to_csv(source_path, index=False)

    fig = make_figure(summary)
    export_figure(fig, args.output_dir, args.stem)
    plt.close(fig)

    print(f"Validated {len(trajectory):,} trajectory rows.")
    print(f"Wrote {len(summary):,} processed Source Data rows to {source_path}.")
    print(f"Exported SVG, PDF, PNG and 600-dpi TIFF to {args.output_dir}.")


if __name__ == "__main__":
    main()
