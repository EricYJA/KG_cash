"""Plot entity composition and top-k coverage as a paper-width 1x4 figure."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

# Full text width of a typical two-column research paper (inches).
FIG_WIDTH = 7.16
FIG_HEIGHT = 2.15

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "ToG-cache" / "output" / "traces"
TRACE_FILES = {
    "WebQSP": TRACE_DIR / "tog_trace_webqsp.json",
    "CWQ": TRACE_DIR / "tog_trace_cwq.json",
}
TRACE_LIMIT = 400
DATASETS = ["WebQSP", "CWQ"]
K_VALUES = [10, 50, 100]

C_UNIQUE = "#2A7F8C"
C_REUSED = "#E07B54"
C_WEBQSP = "#4C72B0"
C_CWQ = "#DD8452"


def load_traces(path):
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def iterative_entity_mentions(traces):
    entity_ids = []
    for trace in traces[:TRACE_LIMIT]:
        for event in trace.get("events", []):
            if event.get("operation") in {"relation_lookup_head", "entity_name_resolve"}:
                entity_id = event.get("input", {}).get("entity_id")
                if entity_id:
                    entity_ids.append(entity_id)
    return entity_ids


def topk_coverage(entity_ids):
    frequencies = Counter(entity_ids).most_common()
    total = len(entity_ids)
    if not total:
        return [0.0] * len(K_VALUES)
    return [sum(count for _, count in frequencies[:k]) / total * 100 for k in K_VALUES]


def draw_composition(ax, unique, reused, title):
    total = unique + reused
    unique_pct = np.divide(unique, total, out=np.zeros_like(unique, dtype=float), where=total > 0) * 100
    reused_pct = np.divide(reused, total, out=np.zeros_like(reused, dtype=float), where=total > 0) * 100
    x = np.arange(len(DATASETS))

    ax.bar(x, unique_pct, 0.55, color=C_UNIQUE, edgecolor="white", linewidth=0.6)
    ax.bar(x, reused_pct, 0.55, bottom=unique_pct, color=C_REUSED, edgecolor="white", linewidth=0.6)
    for i, (up, rp) in enumerate(zip(unique_pct, reused_pct)):
        ax.text(i, up / 2, f"{up:.1f}%", ha="center", va="center", fontsize=7,
                fontweight="bold", color="white")
        ax.text(i, up + rp / 2, f"{rp:.1f}%", ha="center", va="center", fontsize=7,
                fontweight="bold", color="white")

    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    style_axis(ax, title)


def draw_topk(ax, coverage, title):
    for dataset, color, marker in (("WebQSP", C_WEBQSP, "o"), ("CWQ", C_CWQ, "s")):
        values = coverage[dataset]
        ax.plot(K_VALUES, values, marker=marker, color=color, linewidth=1.4,
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.6)
        other_dataset = "CWQ" if dataset == "WebQSP" else "WebQSP"
        for i, (k, value) in enumerate(zip(K_VALUES, values)):
            label_offset = 4 if value >= coverage[other_dataset][i] else -5
            label_va = "bottom" if label_offset > 0 else "top"
            ax.annotate(f"{value:.1f}", (k, value), xytext=(0, label_offset),
                        textcoords="offset points", ha="center", va=label_va,
                        fontsize=6.8, color=color)

    ax.set_xticks(K_VALUES, [f"Top-{k}" for k in K_VALUES])
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    style_axis(ax, title)


def style_axis(ax, title):
    ax.set_title(title, fontsize=8, fontweight="normal", pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# Initial-query statistics retained from the two source plotting scripts.
starting_total = np.array([2_893, 42_134])
starting_unique = np.array([1_601, 10_144])
starting_reused = starting_total - starting_unique
starting_topk = {
    "WebQSP": [7.22, 20.29, 28.72],
    "CWQ": [9.52, 19.08, 25.69],
}

# Compute iterative-traversal statistics once and share them across both plots.
iterative_mentions = {
    dataset: iterative_entity_mentions(load_traces(TRACE_FILES[dataset]))
    for dataset in DATASETS
}
iterative_total = np.array([len(iterative_mentions[dataset]) for dataset in DATASETS])
iterative_unique = np.array([len(set(iterative_mentions[dataset])) for dataset in DATASETS])
iterative_reused = iterative_total - iterative_unique
iterative_topk = {
    dataset: topk_coverage(iterative_mentions[dataset]) for dataset in DATASETS
}


# Nested grids keep within-group spacing independent of the center separator.
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
draw_composition(axes[0], starting_unique, starting_reused, "Initial Query")
draw_composition(axes[1], iterative_unique, iterative_reused, "Iterative Traversal")
draw_topk(axes[2], starting_topk, "Initial Query")
draw_topk(axes[3], iterative_topk, "Iterative Traversal")

# One y-axis scale and one label per two-panel group.
axes[0].sharey(axes[1])
axes[2].sharey(axes[3])
axes[1].tick_params(axis="y", labelleft=False)
axes[3].tick_params(axis="y", labelleft=False)
axes[0].set_ylabel("Entity mentions (%)", fontsize=8, labelpad=2)
axes[2].set_ylabel("Coverage (%)", fontsize=8, labelpad=2)

# Separator spans the plotting region without crossing the shared legend.
separator_ax.set_axis_off()
separator_ax.plot([0.08, 0.08], [0.02, 0.98], transform=separator_ax.transAxes,
                  color="#9a9a9a", linewidth=0.8, clip_on=False)

# One subfigure label per two-panel group; titles distinguish the members.
fig.text(0.27, 0.035, "(a) Entity composition", ha="center", va="center",
         fontsize=8.5, fontweight="bold")
fig.text(0.78, 0.035, "(b) Top-k coverage", ha="center", va="center",
         fontsize=8.5, fontweight="bold")

composition_handles = [
    mpatches.Patch(color=C_UNIQUE, label="Unique"),
    mpatches.Patch(color=C_REUSED, label="Reused"),
]
dataset_handles = [
    plt.Line2D([], [], color=C_WEBQSP, marker="o", label="WebQSP"),
    plt.Line2D([], [], color=C_CWQ, marker="s", label="CWQ"),
]
fig.legend(handles=composition_handles, loc="upper center",
           bbox_to_anchor=(0.27, 1.02), ncol=2, frameon=False, fontsize=7.5)
fig.legend(handles=dataset_handles, loc="upper center",
           bbox_to_anchor=(0.78, 1.02), ncol=2, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.065, right=0.995, bottom=0.18, top=0.78)

output_path = Path(__file__).resolve().parent / "entity_composition_topk_1x4.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {output_path}")
