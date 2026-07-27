import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


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
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="WebQSP",
        zorder=3,
    )
    cwq_line, = ax.plot(
        K_VALUES,
        cwq_coverage,
        "s-",
        color=C_CWQ,
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
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
                fontsize=9.5,
                color=C_WEBQSP,
                fontweight="bold",
            )
            ax.text(
                k,
                cwq + cwq_off,
                f"{cwq:.2f}%",
                ha="center",
                va=cwq_va,
                fontsize=9.5,
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
                fontsize=9.5,
                color=C_WEBQSP,
                fontweight="bold",
            )
            ax.text(
                k,
                cwq + cwq_off,
                f"{cwq:.2f}%",
                ha="center",
                va=cwq_va,
                fontsize=9.5,
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
                fontsize=9.5,
                color=C_WEBQSP,
                fontweight="bold",
            )
            ax.text(
                k,
                cwq - 1.8,
                f"{cwq:.2f}%",
                ha="center",
                va="top",
                fontsize=9.5,
                color=C_CWQ,
                fontweight="bold",
            )

    ax.fill_between(K_VALUES, webqsp_coverage, alpha=0.12, color=C_WEBQSP)
    ax.fill_between(K_VALUES, cwq_coverage, alpha=0.12, color=C_CWQ)

    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(TOP_K_LABELS, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
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
        fontsize=14,
        fontweight="bold",
        color="#222222",
    )
    return webqsp_line, cwq_line


# Starting-entity data from updates.md
starting_topk = {
    "WebQSP": [7.22, 20.29, 28.72],
    "CWQ": [9.52, 19.08, 25.69],
}


# Iterative-entity data from traces
iterative_topk = {}
for dataset in DATASETS:
    traces = load_traces(TRACE_FILES[dataset])
    entity_ids = iterative_entity_mentions(traces)
    iterative_topk[dataset] = topk_coverage(entity_ids, K_VALUES)


# Palette
C_WEBQSP = "#4C72B0"
C_CWQ = "#DD8452"


# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")
ax2.set_facecolor("white")

legend_handles = draw_topk(
    ax1,
    starting_topk["WebQSP"],
    starting_topk["CWQ"],
    subtitle="Initial Query Entities",
    ylabel="Coverage (% of all starting entities)",
)

draw_topk(
    ax2,
    iterative_topk["WebQSP"],
    iterative_topk["CWQ"],
    subtitle="All Entities Considering Iterative Traversal",
    ylabel="Coverage (% of all iterative entities)",
    label_offsets=[(1.2, 1.2), (-1.8, 1.2), (-1.8, 1.2)],
)

fig.legend(
    handles=legend_handles,
    labels=["WebQSP", "CWQ"],
    fontsize=11,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    framealpha=0.85,
    edgecolor="#cccccc",
)

fig.tight_layout(pad=2.5)
fig.subplots_adjust(bottom=0.2)
fig.savefig(
    "entity_topk_coverage.pdf",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved: entity_topk_coverage.pdf")
