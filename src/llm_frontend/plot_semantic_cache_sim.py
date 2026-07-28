from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

THRESHOLD_COLORS = {
    0.99: "#2166ac",
    0.95: "#4393c3",
    0.90: "#92c5de",
    0.85: "#f4a582",
    0.80: "#d6604d",
}
THRESHOLD_MARKERS = {
    0.99: "o",
    0.95: "s",
    0.90: "^",
    0.85: "D",
    0.80: "v",
}
POLICY_TITLES = {
    "semantic_oracle": "Semantic Oracle",
    "semantic_lru":    "Semantic LRU",
    "semantic_lfu":    "Semantic LFU",
}
POLICY_ORDER = ["semantic_oracle", "semantic_lru", "semantic_lfu"]


class _Entry(NamedTuple):
    gain: float     # percentage points above exact-match baseline
    overlap: float  # avg entity Jaccard of semantic hits, already ×100


def load_results(path: Path) -> dict[str, dict[int, dict[float, _Entry]]]:
    """Return {policy: {cache_size: {threshold: _Entry}}}."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    grouped: dict[str, dict[int, dict[float, _Entry]]] = {}
    for r in data.get("question_level_semantic", []):
        if r.get("gain_pct") is None:
            continue
        entry = _Entry(
            gain=r["gain_pct"],
            overlap=r.get("avg_entity_overlap", 0.0) * 100,
        )
        (
            grouped
            .setdefault(r["policy"], {})
            .setdefault(r["cache_size"], {})
        )[r["threshold"]] = entry
    return grouped


def _plot_lines(
    ax: plt.Axes,
    data: dict[int, dict[float, _Entry]],
    metric: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    show_zero_line: bool,
) -> list:
    """Plot one line per threshold; return legend handles."""
    cache_sizes = sorted(data.keys())
    thresholds  = sorted(data[cache_sizes[0]].keys(), reverse=True)  # 0.99 → 0.80

    handles = []
    for threshold in thresholds:
        values = [getattr(data[size][threshold], metric) for size in cache_sizes]
        color  = THRESHOLD_COLORS.get(threshold, "#999999")
        marker = THRESHOLD_MARKERS.get(threshold, "o")
        (line,) = ax.plot(
            cache_sizes, values,
            marker=marker, linewidth=1.8, markersize=6,
            color=color, label=f"≥{threshold:.2f}",
            zorder=3,
        )
        handles.append(line)

    if show_zero_line:
        ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(cache_sizes)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Cache Size", fontsize=19, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=19, labelpad=8, fontweight="bold")
    ax.tick_params(axis="both", labelsize=17)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.margins(y=0.15)
    ax.grid(axis="both", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot semantic cache gain from semantic_cache_summary.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("semantic_cache_summary.json"),
        help="JSON output from run_semantic_cache_sim.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("semantic_cache_sim_gain.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    grouped = load_results(args.input)
    policies = [p for p in POLICY_ORDER if p in grouped]
    if not policies:
        raise SystemExit("No semantic policy data found in input file.")

    n_cols = len(policies)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(7 * n_cols, 9),
        sharex=False,
    )
    fig.patch.set_facecolor("white")

    # axes[row, col]; if n_cols == 1 matplotlib returns 1-D array
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    for row in axes:
        for ax in row:
            ax.set_facecolor("white")

    legend_handles = None

    for col, policy in enumerate(policies):
        data = grouped[policy]

        # ── row 0: gain ──────────────────────────────────────────────
        ax_gain = axes[0, col]
        handles = _plot_lines(
            ax_gain, data, metric="gain",
            ylabel="Hit-Rate Gain (%)" if col == 0 else "",
            ylim=None,
            show_zero_line=True,
        )
        ax_gain.set_title(POLICY_TITLES.get(policy, policy), fontsize=20, fontweight="bold", pad=10)
        if legend_handles is None:
            legend_handles = handles

        # ── row 1: overlap ───────────────────────────────────────────
        ax_ovlp = axes[1, col]
        _plot_lines(
            ax_ovlp, data, metric="overlap",
            ylabel="Avg Entity Overlap (%)" if col == 0 else "",
            ylim=None,
            show_zero_line=False,
        )

    # shared threshold legend at the top
    thresholds_sorted = sorted(THRESHOLD_COLORS.keys(), reverse=True)
    fig.legend(
        handles=legend_handles,
        labels=[f"≥{t:.2f}" for t in thresholds_sorted],
        title="Similarity Threshold",
        title_fontsize=18,
        loc="upper center",
        ncol=len(thresholds_sorted),
        fontsize=17,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle(
        "Semantic Cache: Gain over Exact-Match Baseline & Entity Overlap",
        fontsize=22, fontweight="bold", y=1.06,
    )
    fig.tight_layout(pad=2.5, h_pad=7.0)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
