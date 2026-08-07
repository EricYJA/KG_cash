from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "output" / "cache_sim_summary.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

# Journal (single-column body) sizing: figures are placed at ~6.5in and printed
# at 100%, and the two access patterns stack one per row instead of sitting
# side by side, which no longer fits the narrower text block.
FIG_WIDTH = 6.5
PANEL_HEIGHT = 2.4

POLICY_COLORS = {
    "lru":    "#6A9FD8",
    "lfu":    "#57B38A",
    "fifo":   "#B08FCB",
    "random": "#9AA5B1",
    "belady": "#E8B84B",
    "oracle": "#E8865A",
    # Question-level (semantic) policies, for the RoG summary.
    "exact":           "#7F7F7F",
    "semantic_lru":    "#6A9FD8",
    "semantic_lfu":    "#57B38A",
    "semantic_fifo":   "#B08FCB",
    "semantic_random": "#9AA5B1",
    "semantic_belady": "#E8B84B",
    "semantic_oracle": "#E8865A",
}
POLICY_HATCHES = {
    "lru":    "//",
    "lfu":    "..",
    "fifo":   "\\\\",
    "random": "",
    "belady": "++",
    "oracle": "xx",
    "exact":           "",
    "semantic_lru":    "//",
    "semantic_lfu":    "..",
    "semantic_fifo":   "\\\\",
    "semantic_random": "",
    "semantic_belady": "++",
    "semantic_oracle": "xx",
}
# `Belady (MIN)` is the true offline optimum among demand-paging policies -- it
# still pays every compulsory miss. `Oracle (preload)` instead warms the cache
# with the globally hottest keys before the trace starts, so it dodges those
# compulsory misses and can therefore report a hit rate above the optimum. The
# labels say so, because read as peers the two invite the wrong conclusion.
POLICY_LABELS = {
    "lru":    "LRU",
    "lfu":    "LFU",
    "fifo":   "FIFO",
    "random": "Random",
    "belady": "Belady (MIN)",
    "oracle": "Oracle (preload)",
    "exact":           "Exact",
    "semantic_lru":    "LRU",
    "semantic_lfu":    "LFU",
    "semantic_fifo":   "FIFO",
    "semantic_random": "Random",
    "semantic_belady": "Belady (MIN)",
    "semantic_oracle": "Oracle (gold-gated)",
}


def group_records(records: list[dict]) -> dict[str, dict[int, float]]:
    """Group cache result records into {policy: {cache_size: hit_rate}}."""
    grouped: dict[str, dict[int, float]] = {}
    for record in records:
        grouped.setdefault(record["policy"], {})[record["cache_size"]] = record["hit_rate"]
    return grouped


def load_combined_results(path: Path) -> dict:
    with path.open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if "datasets" not in payload:
        raise ValueError(f"Expected combined cache summary with a 'datasets' field: {path}")
    return payload


def plot_grouped_bars(ax: plt.Axes, results: dict[str, dict[int, float]], title: str) -> tuple:
    policies   = list(results.keys())
    cache_sizes = sorted(next(iter(results.values())).keys())

    n_policies  = len(policies)
    n_sizes     = len(cache_sizes)
    # Keep a group about as wide however many policies it holds, so six bars
    # per group stay legible at the same 6.5in figure width three did.
    bar_width   = min(0.22, 1.32 / n_policies)
    group_gap   = 0.1
    group_width = n_policies * bar_width + group_gap
    x           = np.arange(n_sizes) * group_width

    crowded    = n_policies > 4
    label_size = 4.5 if crowded else 6
    label_rot  = 90 if crowded else 0

    for i, policy in enumerate(policies):
        offsets   = x + i * bar_width - (n_policies - 1) * bar_width / 2
        hit_rates = [results[policy][s] * 100 for s in cache_sizes]
        color  = POLICY_COLORS.get(policy, "#999999")
        hatch  = POLICY_HATCHES.get(policy, "")
        bars = ax.bar(
            offsets, hit_rates, bar_width,
            color=color, alpha=0.85,
            edgecolor="#333333", linewidth=0.8 if crowded else 1.2,
            hatch=hatch, label=POLICY_LABELS.get(policy, policy.upper()),
            zorder=3,
        )
        for bar, rate in zip(bars, hit_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=label_size,
                rotation=label_rot,
                fontweight="bold", color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in cache_sizes], fontsize=9)
    ax.set_xlabel("Cache Size", fontsize=11)
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    # The RoG question-cache summary tops out near 8%, where a fixed 0-110 axis
    # would flatten every bar to nothing. Keep the original headroom whenever the
    # data actually uses the range.
    peak = max(rate for policy in policies for rate in
               (results[policy][s] * 100 for s in cache_sizes))
    ax.set_ylim(0, 110 if peak > 40 else max(peak * 1.55, 5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return ax.get_legend_handles_labels()


def plot_dataset(dataset: str, dataset_results: dict, output_path: Path) -> None:
    sequential_results = group_records(dataset_results["sequential"])
    shuffled_results = group_records(dataset_results["shuffled"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 2 * PANEL_HEIGHT))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    handles, labels = plot_grouped_bars(ax1, sequential_results, "Sequential Access")
    plot_grouped_bars(ax2, shuffled_results, "Shuffled Access")

    # Six long policy names do not fit on one legend row at 6.5in, so wrap to
    # two and give the extra row its headroom.
    ncol = len(labels) if len(labels) <= 4 else (len(labels) + 1) // 2
    legend_rows = -(-len(labels) // ncol)

    fig.suptitle(dataset, fontsize=12, fontweight="bold", y=1.0)
    fig.legend(
        handles,
        labels,
        fontsize=8 if legend_rows > 1 else 9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955 if legend_rows > 1 else 0.945),
        ncol=ncol,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    fig.tight_layout(pad=1.6)
    # Headroom for the suptitle + shared legend that sit above the stacked panels.
    fig.subplots_adjust(top=0.80 if legend_rows == 1 else 0.76, hspace=0.85)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot combined cache simulation results by dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Combined JSON output from cache_simulator.py --combined.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for dataset PDF plots.")
    args = parser.parse_args()

    payload = load_combined_results(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, dataset_results in payload["datasets"].items():
        output_name = f"cache_sim_{dataset.lower()}_comparison.pdf"
        plot_dataset(dataset, dataset_results, args.output_dir / output_name)


if __name__ == "__main__":
    main()
