import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from cache_hit_potential import (
    MENTION_OPS,
    entity_access_stream,
    lfu_hit_rate,
    lru_hit_rate,
)


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

DATASETS = ["WebQSP", "CWQ"]
K_VALUES = [10, 50, 100]
TOP_K_LABELS = [f"Top-{k}" for k in K_VALUES]

# Capacities for the hit-potential panel. Stops at 1000 because both working
# sets (1429 / 929 unique entities) are covered by then, so 2000 and 5000 sit
# flat on the unbounded-cache ceiling and only stretch the axis.
CACHE_SIZES = [10, 50, 100, 200, 500, 1000]


def load_traces(path: Path):
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def iterative_entity_mentions(traces):
    entity_ids = []

    for trace in traces[:TRACE_LIMIT]:
        for event in trace.get("events", []):
            if event.get("operation") == "relation_lookup_head":
                entity_id = event.get("input", {}).get("entity_id")
            elif event.get("operation") == "entity_name_resolve":
                entity_id = event.get("input", {}).get("entity_id")
            else:
                continue

            if entity_id:
                entity_ids.append(entity_id)

    return entity_ids


def topk_coverage(entity_ids, k_values):
    total = len(entity_ids)
    if total == 0:
        return [0.0 for _ in k_values]

    frequencies = Counter(entity_ids).most_common()
    return [
        sum(count for _, count in frequencies[:k]) / total * 100
        for k in k_values
    ]


def draw_topk(ax, webqsp_coverage, cwq_coverage, subtitle, ylabel, label_offsets=None):
    webqsp_line, = ax.plot(
        K_VALUES,
        webqsp_coverage,
        "o-",
        color=C_WEBQSP,
        linewidth=1.8,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="WebQSP",
        zorder=3,
    )
    cwq_line, = ax.plot(
        K_VALUES,
        cwq_coverage,
        "s-",
        color=C_CWQ,
        linewidth=1.8,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="CWQ",
        zorder=3,
    )

    for i, (k, webqsp, cwq) in enumerate(zip(K_VALUES, webqsp_coverage, cwq_coverage)):
        if label_offsets:
            webqsp_off, cwq_off = label_offsets[i]
            webqsp_va = "bottom" if webqsp_off >= 0 else "top"
            cwq_va = "bottom" if cwq_off >= 0 else "top"
            ax.text(
                k,
                webqsp + webqsp_off,
                f"{webqsp:.2f}%",
                ha="center",
                va=webqsp_va,
                fontsize=8,
                color=C_WEBQSP,
                fontweight="bold",
            )
            ax.text(
                k,
                cwq + cwq_off,
                f"{cwq:.2f}%",
                ha="center",
                va=cwq_va,
                fontsize=8,
                color=C_CWQ,
                fontweight="bold",
            )
        elif abs(webqsp - cwq) < 3:
            webqsp_va, cwq_va = ("bottom", "top") if webqsp > cwq else ("top", "bottom")
            webqsp_off, cwq_off = (1.2, -1.2) if webqsp > cwq else (-1.2, 1.2)
            ax.text(
                k,
                webqsp + webqsp_off,
                f"{webqsp:.2f}%",
                ha="center",
                va=webqsp_va,
                fontsize=8,
                color=C_WEBQSP,
                fontweight="bold",
            )
            ax.text(
                k,
                cwq + cwq_off,
                f"{cwq:.2f}%",
                ha="center",
                va=cwq_va,
                fontsize=8,
                color=C_CWQ,
                fontweight="bold",
            )
        else:
            ax.text(
                k,
                webqsp + 1.2,
                f"{webqsp:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=C_WEBQSP,
                fontweight="bold",
            )
            ax.text(
                k,
                cwq - 1.8,
                f"{cwq:.2f}%",
                ha="center",
                va="top",
                fontsize=8,
                color=C_CWQ,
                fontweight="bold",
            )

    ax.fill_between(K_VALUES, webqsp_coverage, alpha=0.12, color=C_WEBQSP)
    ax.fill_between(K_VALUES, cwq_coverage, alpha=0.12, color=C_CWQ)

    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(TOP_K_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
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
    return webqsp_line, cwq_line


def draw_hit_potential(ax, hit_potential, subtitle, ylabel):
    """Panel 3: how much of the reuse a bounded cache can actually convert.

    Coverage and reuse count repeats anywhere in the workload; this replays the
    same access stream through LRU and LFU so eviction distance is priced in.
    Colour keeps the dataset identity used above, and the line style (solid /
    dashed, filled / hollow markers) carries the policy.
    """
    styles = {
        "WebQSP": {"color": C_WEBQSP, "marker": "o"},
        "CWQ": {"color": C_CWQ, "marker": "s"},
    }
    policies = {
        "lru": {"linestyle": "-", "fill": True},
        "lfu": {"linestyle": "--", "fill": False},
    }

    for dataset in DATASETS:
        style = styles[dataset]
        series = hit_potential[dataset]

        # Unbounded-cache ceiling: the reuse percentage the panel above reports.
        ax.axhline(
            series["reuse_pct"],
            color=style["color"],
            linewidth=0.9,
            linestyle=":",
            alpha=0.55,
            zorder=1,
        )

        for policy, policy_style in policies.items():
            ax.plot(
                CACHE_SIZES,
                series[policy],
                linestyle=policy_style["linestyle"],
                marker=style["marker"],
                color=style["color"],
                linewidth=1.8,
                markersize=5,
                markerfacecolor=style["color"] if policy_style["fill"] else "white",
                markeredgecolor=style["color"] if not policy_style["fill"] else "white",
                markeredgewidth=1.0,
                zorder=3,
            )

    # Direct-label only what carries the message: what a 10-entry cache already
    # returns under LRU, at the empty left edge.
    for dataset, lru_offset in (("WebQSP", 2.5), ("CWQ", -3.0)):
        style = styles[dataset]
        ax.text(
            CACHE_SIZES[0],
            hit_potential[dataset]["lru"][0] + lru_offset,
            f"{hit_potential[dataset]['lru'][0]:.1f}%",
            ha="left",
            va="bottom" if lru_offset >= 0 else "top",
            fontsize=8,
            color=style["color"],
            fontweight="bold",
        )

    # The two ceilings are only ~5pp apart, so labelling each dotted line in
    # place collides with the curves. One row in the empty band up top instead.
    parts = [("Unbounded ceiling:", "#666666", "normal")] + [
        (f"{hit_potential[d]['reuse_pct']:.1f}%", styles[d]["color"], "bold")
        for d in DATASETS
    ]
    # Lay the row out left to right by measuring each piece, so the segments
    # never overlap regardless of the rendered font.
    x_pos = 0.015
    for text, color, weight in parts:
        drawn = ax.text(
            x_pos,
            0.94,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8,
            color=color,
            fontweight=weight,
        )
        ax.figure.canvas.draw()
        extent = drawn.get_window_extent().transformed(ax.transAxes.inverted())
        x_pos = extent.x1 + 0.015

    policy_handles = [
        plt.Line2D([], [], color="#666666", linestyle="-", linewidth=1.8, label="LRU"),
        plt.Line2D([], [], color="#666666", linestyle="--", linewidth=1.8, label="LFU"),
        plt.Line2D([], [], color="#666666", linestyle=":", linewidth=1.0,
                   label="Unbounded"),
    ]
    ax.legend(
        handles=policy_handles,
        fontsize=8,
        loc="lower right",
        handlelength=2.6,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    ax.set_xscale("log")
    ax.set_xticks(CACHE_SIZES)
    ax.set_xticklabels([str(size) for size in CACHE_SIZES], fontsize=9)
    ax.minorticks_off()
    ax.set_xlim(CACHE_SIZES[0] * 0.9, CACHE_SIZES[-1] * 1.12)
    ax.set_xlabel("Cache capacity (entries)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.5,
        -0.34,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#222222",
    )


# Starting-entity data from updates.md
starting_topk = {
    "WebQSP": [7.22, 20.29, 28.72],
    "CWQ": [9.52, 19.08, 25.69],
}


# Iterative-entity data from traces
iterative_topk = {}
# Cache hit potential over that same access stream (offline replay, no LLM).
hit_potential = {}
for dataset in DATASETS:
    traces = load_traces(TRACE_FILES[dataset])
    entity_ids = iterative_entity_mentions(traces)
    iterative_topk[dataset] = topk_coverage(entity_ids, K_VALUES)

    stream = entity_access_stream(traces[:TRACE_LIMIT], MENTION_OPS)
    unique = len(set(stream))
    hit_potential[dataset] = {
        "reuse_pct": (len(stream) - unique) / len(stream) * 100,
        "lru": [lru_hit_rate(stream, size) for size in CACHE_SIZES],
        "lfu": [lfu_hit_rate(stream, size) for size in CACHE_SIZES],
    }


# Palette
C_WEBQSP = "#4C72B0"
C_CWQ = "#DD8452"


# Plot
# Stacked one panel per row: at a single-column journal width the two views
# do not fit side by side.
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(FIG_WIDTH, 3 * PANEL_HEIGHT))
fig.patch.set_facecolor("white")
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("white")

legend_handles = draw_topk(
    ax1,
    starting_topk["WebQSP"],
    starting_topk["CWQ"],
    subtitle="Initial Query Entities",
    ylabel="Coverage (%)",
)

draw_topk(
    ax2,
    iterative_topk["WebQSP"],
    iterative_topk["CWQ"],
    subtitle="All Entities Considering Iterative Traversal",
    ylabel="Coverage (%)",
    label_offsets=[(1.2, 1.2), (-1.8, 1.2), (-1.8, 1.2)],
)

draw_hit_potential(
    ax3,
    hit_potential,
    subtitle="Cache Hit Potential Over the Same Access Stream",
    ylabel="Hit rate (%)",
)

fig.legend(
    handles=legend_handles,
    labels=["WebQSP", "CWQ"],
    fontsize=9,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    framealpha=0.85,
    edgecolor="#cccccc",
)

fig.tight_layout(pad=1.6)
# Room under each panel for its bold subtitle, and at the top for the legend.
fig.subplots_adjust(bottom=0.09, top=0.95, hspace=0.75)
fig.savefig(
    "entity_topk_coverage.pdf",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved: entity_topk_coverage.pdf")
