import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Journal (single-column body) sizing: the figure is placed at ~6.5in and
# printed at 100%, so tick labels sit at body-text size rather than the
# inflated sizes an IEEE two-column shrink needed.
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})

FIG_WIDTH = 6.5      # single-column journal text width, in inches
PANEL_HEIGHT = 2.5   # per stacked panel

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "ToG-cache" / "output" / "traces"
TRACE_FILES = {
    "WebQSP": TRACE_DIR / "tog_trace_webqsp.json",
    "CWQ": TRACE_DIR / "tog_trace_cwq.json",
}
TRACE_LIMIT = 400


def load_traces(path: Path):
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def iterative_entity_counts(traces):
    seen = set()
    total = 0
    unique = 0

    for trace in traces[:TRACE_LIMIT]:
        for event in trace.get("events", []):
            if event.get("operation") == "relation_lookup_head":
                entity_id = event.get("input", {}).get("entity_id")
            elif event.get("operation") == "entity_name_resolve":
                entity_id = event.get("input", {}).get("entity_id")
            else:
                continue

            if not entity_id:
                continue

            total += 1
            if entity_id not in seen:
                seen.add(entity_id)
                unique += 1

    reused = total - unique
    return total, unique, reused


def draw_stacked(ax, datasets, unique_pct, reused_pct, unique_counts, reused_counts, total_counts, subtitle, ylabel):
    x = np.arange(len(datasets))
    width = 0.45

    ax.bar(
        x,
        unique_pct,
        width,
        color=C_UNIQUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.bar(
        x,
        reused_pct,
        width,
        bottom=unique_pct,
        color=C_REUSED,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for i, (up, rp, u, r, t) in enumerate(zip(unique_pct, reused_pct, unique_counts, reused_counts, total_counts)):
        if up > 0:
            ax.text(
                x[i],
                up / 2,
                f"{up:.1f}%\n({u:,})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
        if rp > 0:
            ax.text(
                x[i],
                up + rp / 2,
                f"{rp:.1f}%\n({r:,})",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
        ax.text(
            x[i],
            101.5,
            f"Total: {t:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5,
        -0.2,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )


# Starting-entity data from updates.md
datasets = ["WebQSP", "CWQ"]
starting_total = np.array([2_893, 42_134])
starting_unique = np.array([1_601, 10_144])
starting_reused = starting_total - starting_unique
starting_unique_pct = starting_unique / starting_total * 100
starting_reused_pct = starting_reused / starting_total * 100


# Iterative-entity data from traces
iterative_total = []
iterative_unique = []
iterative_reused = []
for dataset in datasets:
    traces = load_traces(TRACE_FILES[dataset])
    total, unique, reused = iterative_entity_counts(traces)
    iterative_total.append(total)
    iterative_unique.append(unique)
    iterative_reused.append(reused)

iterative_total = np.array(iterative_total)
iterative_unique = np.array(iterative_unique)
iterative_reused = np.array(iterative_reused)
iterative_unique_pct = np.divide(iterative_unique, iterative_total, out=np.zeros_like(iterative_unique, dtype=float), where=iterative_total > 0) * 100
iterative_reused_pct = np.divide(iterative_reused, iterative_total, out=np.zeros_like(iterative_reused, dtype=float), where=iterative_total > 0) * 100


# Palette
C_UNIQUE = "#2A7F8C"
C_REUSED = "#E07B54"


# Plot
# Stacked one panel per row: at a single-column journal width the two views
# do not fit side by side.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 2 * PANEL_HEIGHT))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")
ax2.set_facecolor("white")

draw_stacked(
    ax1,
    datasets,
    starting_unique_pct,
    starting_reused_pct,
    starting_unique,
    starting_reused,
    starting_total,
    subtitle="Initial Query Entities",
    ylabel="Entity Percentage",
)

draw_stacked(
    ax2,
    datasets,
    iterative_unique_pct,
    iterative_reused_pct,
    iterative_unique,
    iterative_reused,
    iterative_total,
    subtitle=f"All Entities Considering Iterative Traversal",
    ylabel="Entity Percentage",
)

legend_handles = [
    mpatches.Patch(color=C_UNIQUE, label="Unique"),
    mpatches.Patch(color=C_REUSED, label="Reused"),
]
fig.legend(
    handles=legend_handles,
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    framealpha=0.85,
    edgecolor="#cccccc",
)

fig.tight_layout(pad=1.6)
# Room under each panel for its bold subtitle, and at the top for the legend.
fig.subplots_adjust(bottom=0.11, top=0.93, hspace=0.55)
fig.savefig(
    "entity_unique_vs_reused_mentions.pdf",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved: entity_unique_vs_reused_mentions.pdf")
