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
# Journal (single-column body) sizing: the figure is placed at ~6.5in and printed
# at 100%, so fonts are body-text sized rather than inflated for a two-column
# shrink, and panels stack one per row.
FIG_WIDTH = 6.5
PANEL_HEIGHT = 2.4

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
            marker=marker, linewidth=1.6, markersize=5,
            color=color, label=f"≥{threshold:.2f}",
            zorder=3,
        )
        handles.append(line)

    if show_zero_line:
        ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(cache_sizes)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Cache Size", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, labelpad=6, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9)
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
        help="Output base path; one file per policy is written to "
             "<stem>_<policy><suffix>.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    grouped = load_results(args.input)
    policies = [p for p in POLICY_ORDER if p in grouped]
    if not policies:
        raise SystemExit("No semantic policy data found in input file.")

    for policy in policies:
        path = args.output.with_name(
            f"{args.output.stem}_{policy}{args.output.suffix}")
        _plot_policy(grouped[policy], policy, path)


def _plot_policy(data: dict[int, dict[float, _Entry]], policy: str, output: Path) -> None:
    """One journal-width figure for a single policy: gain over overlap, stacked.

    Each policy gets its own file rather than a column of a wide grid: at a
    single-column journal width the policies do not fit side by side, and stacking
    every policy into one figure would run several pages tall.
    """
    fig, (ax_gain, ax_ovlp) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH, 2 * PANEL_HEIGHT),
        sharex=False,
    )
    fig.patch.set_facecolor("white")
    for ax in (ax_gain, ax_ovlp):
        ax.set_facecolor("white")

    legend_handles = _plot_lines(
        ax_gain, data, metric="gain",
        # Stacked panels share no left column, so every row is labelled.
        ylabel="Hit-Rate Gain (%)",
        ylim=None,
        show_zero_line=True,
    )
    _plot_lines(
        ax_ovlp, data, metric="overlap",
        ylabel="Avg Entity Overlap (%)",
        ylim=None,
        show_zero_line=False,
    )

    # shared threshold legend at the top
    thresholds_sorted = sorted(THRESHOLD_COLORS.keys(), reverse=True)
    fig.legend(
        handles=legend_handles,
        labels=[f"≥{t:.2f}" for t in thresholds_sorted],
        title="Similarity Threshold",
        title_fontsize=10,
        loc="upper center",
        ncol=len(thresholds_sorted),
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 1.005),
    )

    fig.suptitle(
        f"{POLICY_TITLES.get(policy, policy)}: Gain over Exact-Match Baseline "
        f"& Entity Overlap",
        fontsize=12, fontweight="bold", y=1.06,
    )
    fig.tight_layout(pad=1.6, h_pad=2.6)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
