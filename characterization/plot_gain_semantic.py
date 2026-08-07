"""Paper-width 1x4 characterization plot for semantic cache behavior."""

from pathlib import Path

import matplotlib.pyplot as plt


FIG_WIDTH = 7.16
FIG_HEIGHT = 2.35

THRESHOLDS = ["0.99", "0.95", "0.90", "0.85", "0.80"]
CACHE_SIZES = [10, 50, 100, 200, 500]

WEBQSP_GAIN = {
    10: [0.00, 0.00, 0.00, 0.04, 0.07],
    50: [0.00, 0.14, 0.25, 0.39, 0.74],
    100: [0.07, 0.32, 0.99, 1.73, 2.37],
    200: [0.11, 0.64, 1.98, 3.08, 4.25],
    500: [0.21, 1.31, 3.68, 5.73, 7.82],
}
CWQ_GAIN = {
    10: [0.00, 0.04, 0.08, 0.17, 0.29],
    50: [0.00, 0.06, 0.20, 0.41, 0.65],
    100: [0.00, 0.10, 0.30, 0.66, 1.28],
    200: [0.01, 0.19, 0.54, 1.10, 2.18],
    500: [0.02, 0.38, 1.23, 2.48, 4.50],
}

WEBQSP_OVERLAP = {
    "0.99": [0.00, 0.00, 100.00, 100.00, 100.00],
    "0.95": [0.00, 100.00, 100.00, 100.00, 100.00],
    "0.90": [0.00, 71.43, 87.50, 91.96, 96.63],
    "0.85": [100.00, 81.82, 90.82, 93.68, 95.99],
    "0.80": [100.00, 85.71, 91.79, 94.58, 95.25],
}
CWQ_OVERLAP = {
    "0.99": [100.00, 100.00, 100.00, 100.00, 86.67],
    "0.95": [100.00, 97.92, 96.91, 94.55, 94.87],
    "0.90": [95.24, 87.50, 90.45, 88.94, 86.88],
    "0.85": [86.96, 77.53, 78.60, 78.02, 77.39],
    "0.80": [70.00, 65.33, 67.00, 65.15, 67.29],
}

MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", "-"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
})


def style_axis(ax, title):
    ax.set_title(title, fontsize=8, fontweight="normal", pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def draw_gain(ax, values, title):
    for i, cache_size in enumerate(CACHE_SIZES):
        ax.plot(THRESHOLDS, values[cache_size], marker=MARKERS[i],
                linestyle=LINESTYLES[i], color=COLORS[i], linewidth=1.2,
                markersize=3.2, label=str(cache_size))
    ax.set_ylim(0, 8.5)
    style_axis(ax, title)


def draw_overlap(ax, values, title):
    for i, threshold in enumerate(THRESHOLDS):
        ax.plot(CACHE_SIZES, values[threshold], marker=MARKERS[i],
                linestyle=LINESTYLES[i], color=COLORS[i], linewidth=1.2,
                markersize=3.2, label=threshold)
    ax.set_xticks([10, 100, 200, 500])
    ax.set_ylim(0, 105)
    style_axis(ax, title)


fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
outer_grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.18, 1], wspace=0.02)
left_grid = outer_grid[0, 0].subgridspec(1, 2, wspace=0.25)
right_grid = outer_grid[0, 2].subgridspec(1, 2, wspace=0.25)
axes = [
    fig.add_subplot(left_grid[0, 0]),
    fig.add_subplot(left_grid[0, 1]),
    fig.add_subplot(right_grid[0, 0]),
    fig.add_subplot(right_grid[0, 1]),
]
separator_ax = fig.add_subplot(outer_grid[0, 1])

draw_overlap(axes[0], WEBQSP_OVERLAP, "WebQSP")
draw_overlap(axes[1], CWQ_OVERLAP, "CWQ")
draw_gain(axes[2], WEBQSP_GAIN, "WebQSP")
draw_gain(axes[3], CWQ_GAIN, "CWQ")

axes[0].sharey(axes[1])
axes[2].sharey(axes[3])
axes[1].tick_params(axis="y", labelleft=False)
axes[3].tick_params(axis="y", labelleft=False)
axes[0].set_ylabel("Average Entity Overlap (%)", fontsize=8, labelpad=2)
axes[2].set_ylabel("Hit-rate Gain (%)", fontsize=8, labelpad=2)

separator_ax.set_axis_off()
separator_ax.plot([0.28, 0.28], [0.02, 0.98], transform=separator_ax.transAxes,
                  color="#9a9a9a", linewidth=0.8, clip_on=False)

overlap_handles, _ = axes[0].get_legend_handles_labels()
gain_handles, _ = axes[2].get_legend_handles_labels()
fig.legend(overlap_handles, THRESHOLDS, title="Threshold",
           loc="upper center", bbox_to_anchor=(0.27, 1.025), ncol=5,
           frameon=False, fontsize=7, title_fontsize=7.5)
fig.legend(gain_handles, [str(size) for size in CACHE_SIZES], title="Cache size",
           loc="upper center", bbox_to_anchor=(0.78, 1.025), ncol=5,
           frameon=False, fontsize=7, title_fontsize=7.5)

fig.text(0.27, 0.105, "Cache size", ha="center", fontsize=7.5)
fig.text(0.78, 0.105, "Similarity threshold", ha="center", fontsize=7.5)
fig.text(0.27, 0.02, "(a) Semantic-hit entity overlap", ha="center",
         fontsize=8.5, fontweight="bold")
fig.text(0.78, 0.02, "(b) Hit-rate improvement", ha="center",
         fontsize=8.5, fontweight="bold")

fig.subplots_adjust(left=0.065, right=0.995, bottom=0.24, top=0.76)

output_path = Path(__file__).resolve().parent / "gain_semantic_1x4.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {output_path}")
