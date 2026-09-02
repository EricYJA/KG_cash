"""Plot entity composition, top-k coverage, and cache hit potential as a
paper-width 1x5 figure."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from cache_hit_potential import lfu_hit_rate, lru_hit_rate


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
# Capacities for the hit-potential panel. Stops at 1000 because both working
# sets are covered by then, so larger caches sit flat on the reuse ceiling.
CACHE_SIZES = [10, 50, 100, 200, 500, 1000]
CACHE_TICKS = [10, 100, 1000]  # only three fit at this panel width

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

    # Wider bars and no "%" on the in-bar labels: at five panels the text is
    # otherwise wider than the bar it sits in. The y axis carries the unit.
    ax.bar(x, unique_pct, 0.7, color=C_UNIQUE, edgecolor="white", linewidth=0.6)
    ax.bar(x, reused_pct, 0.7, bottom=unique_pct, color=C_REUSED, edgecolor="white", linewidth=0.6)
    for i, (up, rp) in enumerate(zip(unique_pct, reused_pct)):
        ax.text(i, up / 2, f"{up:.1f}", ha="center", va="center", fontsize=7,
                fontweight="bold", color="white")
        ax.text(i, up + rp / 2, f"{rp:.1f}", ha="center", va="center", fontsize=7,
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

    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    style_axis(ax, title)


def draw_hit_potential(ax, hit_potential, title):
    """How much of the reuse above a bounded cache actually converts.

    Composition and coverage count repeats anywhere in the workload; this
    replays the same mention stream through LRU and LFU so eviction distance
    is priced in. Colour keeps the dataset identity, line style the policy.
    """
    for dataset, color, marker in (("WebQSP", C_WEBQSP, "o"), ("CWQ", C_CWQ, "s")):
        series = hit_potential[dataset]
        # Unbounded-cache ceiling: the reused share panel (a) reports.
        ax.axhline(series["reuse_pct"], color=color, linewidth=0.7,
                   linestyle=":", alpha=0.55, zorder=1)
        for policy, linestyle, filled in (("lru", "-", True), ("lfu", "--", False)):
            ax.plot(CACHE_SIZES, series[policy], linestyle=linestyle, marker=marker,
                    color=color, linewidth=1.4, markersize=3.5,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor="white" if filled else color,
                    markeredgewidth=0.6, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(CACHE_TICKS, [str(size) for size in CACHE_TICKS])
    ax.minorticks_off()
    ax.set_ylim(0, 80)
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
# Cache hit potential over that same mention stream (offline replay, no LLM).
hit_potential = {
    dataset: {
        "reuse_pct": (total - unique) / total * 100,
        "lru": [lru_hit_rate(iterative_mentions[dataset], size) for size in CACHE_SIZES],
        "lfu": [lfu_hit_rate(iterative_mentions[dataset], size) for size in CACHE_SIZES],
    }
    for dataset, total, unique in zip(DATASETS, iterative_total, iterative_unique)
}


# Nested grids keep within-group spacing independent of the separators. The
# third group holds a single panel, so it gets a narrower slot than the pairs.
GROUP_WIDTHS = [1, 0.26, 1, 0.26, 0.62]
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
outer_grid = fig.add_gridspec(1, 5, width_ratios=GROUP_WIDTHS, wspace=0.02)
left_grid = outer_grid[0, 0].subgridspec(1, 2, wspace=0.25)
middle_grid = outer_grid[0, 2].subgridspec(1, 2, wspace=0.25)
axes = [
    fig.add_subplot(left_grid[0, 0]),
    fig.add_subplot(left_grid[0, 1]),
    fig.add_subplot(middle_grid[0, 0]),
    fig.add_subplot(middle_grid[0, 1]),
    fig.add_subplot(outer_grid[0, 4]),
]
separator_axes = [fig.add_subplot(outer_grid[0, 1]), fig.add_subplot(outer_grid[0, 3])]
draw_composition(axes[0], starting_unique, starting_reused, "Initial Query")
draw_composition(axes[1], iterative_unique, iterative_reused, "Iterative Traversal")
draw_topk(axes[2], starting_topk, "Initial Query")
draw_topk(axes[3], iterative_topk, "Iterative Traversal")
draw_hit_potential(axes[4], hit_potential, "Iterative Traversal")

# One y-axis scale and one label per two-panel group.
axes[0].sharey(axes[1])
axes[2].sharey(axes[3])
axes[1].tick_params(axis="y", labelleft=False)
axes[3].tick_params(axis="y", labelleft=False)
axes[0].set_ylabel("Entity mentions (%)", fontsize=8, labelpad=2)
axes[2].set_ylabel("Coverage (%)", fontsize=8, labelpad=2)
axes[4].set_ylabel("Hit rate (%)", fontsize=8, labelpad=2)

# Separators span the plotting region without crossing the shared legends.
for separator_ax in separator_axes:
    separator_ax.set_axis_off()
    separator_ax.plot([0.08, 0.08], [0.02, 0.98], transform=separator_ax.transAxes,
                      color="#9a9a9a", linewidth=0.8, clip_on=False)

# One subfigure label per group; the panel titles distinguish the members.
FIG_LEFT, FIG_RIGHT = 0.065, 0.995


def group_center(start_index, end_index):
    """Figure-fraction center of the outer-grid columns [start, end)."""
    total = sum(GROUP_WIDTHS)
    offset = sum(GROUP_WIDTHS[:start_index]) + sum(GROUP_WIDTHS[start_index:end_index]) / 2
    return FIG_LEFT + offset / total * (FIG_RIGHT - FIG_LEFT)


group_centers = [group_center(0, 1), group_center(2, 3), group_center(4, 5)]
group_labels = [
    "(a) Entity composition",
    "(b) Top-k coverage",
    "(c) Hit potential vs. cache size",
]
for center, label in zip(group_centers, group_labels):
    fig.text(center, 0.035, label, ha="center", va="center",
             fontsize=8.5, fontweight="bold")

composition_handles = [
    mpatches.Patch(color=C_UNIQUE, label="Unique"),
    mpatches.Patch(color=C_REUSED, label="Reused"),
]
dataset_handles = [
    plt.Line2D([], [], color=C_WEBQSP, marker="o", label="WebQSP"),
    plt.Line2D([], [], color=C_CWQ, marker="s", label="CWQ"),
]
# Policy legend for (c); dataset colours are already named by (b)'s legend.
policy_handles = [
    plt.Line2D([], [], color="#666666", linestyle="-", label="LRU"),
    plt.Line2D([], [], color="#666666", linestyle="--", label="LFU"),
    plt.Line2D([], [], color="#666666", linestyle=":", linewidth=0.9,
               label="Unbounded"),
]
fig.legend(handles=composition_handles, loc="upper center",
           bbox_to_anchor=(group_centers[0], 1.02), ncol=2, frameon=False, fontsize=7.5)
fig.legend(handles=dataset_handles, loc="upper center",
           bbox_to_anchor=(group_centers[1], 1.02), ncol=2, frameon=False, fontsize=7.5)
fig.legend(handles=policy_handles, loc="upper center",
           bbox_to_anchor=(group_centers[2], 1.02), ncol=3, frameon=False,
           fontsize=7.5, handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
fig.subplots_adjust(left=FIG_LEFT, right=FIG_RIGHT, bottom=0.18, top=0.78)

# Filename kept from the 1x4 version so existing \includegraphics keeps working.
output_path = Path(__file__).resolve().parent / "entity_composition_topk_1x4.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {output_path}")
