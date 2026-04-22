from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POLICY_COLORS = {
    "lru":    "#6A9FD8",
    "lfu":    "#57B38A",
    "oracle": "#E8865A",
}
POLICY_HATCHES = {
    "lru":    "//",
    "lfu":    "..",
    "oracle": "xx",
}
POLICY_LABELS = {
    "lru":    "LRU",
    "lfu":    "LFU",
    "oracle": "Oracle",
}


def load_results(path: Path) -> dict[str, dict[int, float]]:
    """Load JSON results into {policy: {cache_size: hit_rate}}."""
    with path.open(encoding="utf-8") as f:
        records = json.load(f)
    grouped: dict[str, dict[int, float]] = {}
    for r in records:
        grouped.setdefault(r["policy"], {})[r["cache_size"]] = r["hit_rate"]
    return grouped


def plot_grouped_bars(ax: plt.Axes, results: dict[str, dict[int, float]], title: str) -> None:
    policies   = list(results.keys())
    cache_sizes = sorted(next(iter(results.values())).keys())

    n_policies  = len(policies)
    n_sizes     = len(cache_sizes)
    bar_width   = 0.22
    group_gap   = 0.1
    group_width = n_policies * bar_width + group_gap
    x           = np.arange(n_sizes) * group_width

    for i, policy in enumerate(policies):
        offsets   = x + i * bar_width - (n_policies - 1) * bar_width / 2
        hit_rates = [results[policy][s] * 100 for s in cache_sizes]
        color  = POLICY_COLORS.get(policy, "#999999")
        hatch  = POLICY_HATCHES.get(policy, "")
        bars = ax.bar(
            offsets, hit_rates, bar_width,
            color=color, alpha=0.85,
            edgecolor="#333333", linewidth=1.2,
            hatch=hatch, label=POLICY_LABELS.get(policy, policy.upper()),
            zorder=3,
        )
        for bar, rate in zip(bars, hit_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=6.5,
                fontweight="bold", color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in cache_sizes], fontsize=10)
    ax.set_xlabel("Cache Size", fontsize=11)
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cache simulation results from two JSON files."
    )
    parser.add_argument("--sequential", type=Path, default="direct_cache_summary_no_shuffle.json",
                        help="JSON results from sequential (non-shuffled) simulation.")
    parser.add_argument("--shuffled",   type=Path, default="direct_cache_summary_shuffle.json",
                        help="JSON results from shuffled simulation.")
    parser.add_argument("--output",     type=Path, default="cache_sim_comparison_direct.png",
                        help="Output PNG path.")
    args = parser.parse_args()

    seq_results  = load_results(args.sequential)
    shuf_results = load_results(args.shuffled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    plot_grouped_bars(ax1, seq_results,  "Sequential Access")
    plot_grouped_bars(ax2, shuf_results, "Shuffled Access")

    fig.tight_layout(pad=2.5)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
