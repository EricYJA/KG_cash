"""Plot WebQSP and CWQ cache simulations in one paper-width 1x4 figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "output" / "cache_sim_summary.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output" / "cache_sim_1x4.pdf"

FIG_WIDTH = 7.16
FIG_HEIGHT = 2.45

POLICY_COLORS = {
    "lru": "#6A9FD8",
    "lfu": "#57B38A",
    "fifo": "#B08FCB",
    "random": "#9AA5B1",
    "belady": "#E8B84B",
    "oracle": "#E8865A",
}
POLICY_HATCHES = {
    "lru": "//",
    "lfu": "..",
    "fifo": "\\\\",
    "random": "",
    "belady": "++",
    "oracle": "xx",
}
POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "fifo": "FIFO",
    "random": "Random",
    "belady": "Belady (MIN)",
    "oracle": "Oracle",
}

plt.rcParams.update({
    "font.size": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
})


def load_results(path: Path) -> dict:
    with path.open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if "datasets" not in payload:
        raise ValueError(f"Expected a combined cache summary: {path}")
    return payload["datasets"]


def group_records(records: list[dict]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, float]] = {}
    for record in records:
        grouped.setdefault(record["policy"], {})[record["cache_size"]] = record["hit_rate"]
    return grouped


def draw_bars(ax: plt.Axes, records: list[dict], title: str):
    results = group_records(records)
    policies = list(results)
    cache_sizes = sorted(next(iter(results.values())))
    x = np.arange(len(cache_sizes))
    bar_width = min(0.24, 0.74 / len(policies))

    for i, policy in enumerate(policies):
        offset = (i - (len(policies) - 1) / 2) * bar_width
        rates = [results[policy][size] * 100 for size in cache_sizes]
        bars = ax.bar(
            x + offset,
            rates,
            bar_width,
            color=POLICY_COLORS.get(policy, "#999999"),
            edgecolor="#333333",
            linewidth=0.6,
            hatch=POLICY_HATCHES.get(policy, ""),
            label=POLICY_LABELS.get(policy, policy.upper()),
            zorder=3,
        )

    ax.set_xticks(x, [str(size) for size in cache_sizes])
    ax.set_xlabel("Cache Size", fontsize=7.5, labelpad=3)
    ax.set_ylim(0, 70)
    ax.set_yticks(np.arange(0, 71, 10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_title(title, fontsize=8, fontweight="normal", pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return ax.get_legend_handles_labels()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paper-width 1x4 cache-simulation figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = load_results(args.input)
    missing = {"WebQSP", "CWQ"} - datasets.keys()
    if missing:
        raise ValueError(f"Missing required datasets: {', '.join(sorted(missing))}")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    outer_grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.08, 1], wspace=0.01)
    left_grid = outer_grid[0, 0].subgridspec(1, 2, wspace=0.18)
    right_grid = outer_grid[0, 2].subgridspec(1, 2, wspace=0.18)
    axes = [
        fig.add_subplot(left_grid[0, 0]),
        fig.add_subplot(left_grid[0, 1]),
        fig.add_subplot(right_grid[0, 0]),
        fig.add_subplot(right_grid[0, 1]),
    ]
    separator_ax = fig.add_subplot(outer_grid[0, 1])

    handles, labels = draw_bars(axes[0], datasets["WebQSP"]["sequential"], "Sequential Access")
    draw_bars(axes[1], datasets["WebQSP"]["shuffled"], "Shuffled Access")
    draw_bars(axes[2], datasets["CWQ"]["sequential"], "Sequential Access")
    draw_bars(axes[3], datasets["CWQ"]["shuffled"], "Shuffled Access")

    for ax in axes[1:]:
        ax.sharey(axes[0])
        ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].set_ylabel("Hit Rate (%)", fontsize=8, labelpad=2)

    separator_ax.set_axis_off()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(labels),
        frameon=False,
        fontsize=7.5,
    )
    fig.text(0.27, 0.02, "(a) WebQSP", ha="center", fontsize=8.5, fontweight="bold")
    fig.text(0.78, 0.02, "(b) CWQ", ha="center", fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.80)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
