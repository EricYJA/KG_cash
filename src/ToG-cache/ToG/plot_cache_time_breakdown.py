from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Journal (single-column body) sizing: the figure is placed at ~6.5in and
# printed at 100%, so labels sit at body-text size rather than the inflated
# sizes an IEEE two-column shrink needed.
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})

FIG_WIDTH = 6.5        # single-column journal text width, in inches
DATASET_HEIGHT = 3.0   # per dataset (its total + zoomed KG rows)

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "output" / "cache_sim_summary.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "fifo": "FIFO",
    "random": "Random",
    "belady": "Belady",
    "oracle": "Oracle*",
    "exact": "Exact",
    "semantic_lru": "LRU",
    "semantic_lfu": "LFU",
    "semantic_fifo": "FIFO",
    "semantic_random": "Random",
    "semantic_belady": "Belady",
    "semantic_oracle": "Oracle*",
}
# Preferred left-to-right order; anything else in the summary is appended after.
POLICY_ORDER = [
    "exact", "lru", "semantic_lru", "lfu", "semantic_lfu",
    "fifo", "semantic_fifo", "random", "semantic_random",
    "belady", "semantic_belady", "oracle", "semantic_oracle",
]
# The cached stage is drawn in blue, the stage the cache cannot touch in green.
BREAKDOWN_COLORS = {"cached": "#4C72B0", "uncached": "#55A868"}
# Older summaries predate metadata.stage_keys; they were all ToG KG/LLM runs.
DEFAULT_STAGE_KEYS = {"cached": "kg", "uncached": "llm"}
DEFAULT_STAGE_LABELS = {"cached": "KG", "uncached": "LLM"}
SPEEDUP_COLOR = "#C44E52"


def stage_config(payload: dict) -> tuple[dict, dict]:
    """Which time bucket the cache shrinks, and which it leaves alone.

    ToG caches KG requests and leaves the LLM calls alone; RoG caches the
    planner LLM call and leaves the reasoner alone. Both summaries name their
    buckets in metadata so this plotter works on either.
    """
    metadata = payload.get("metadata", {})
    keys = {**DEFAULT_STAGE_KEYS, **metadata.get("stage_keys", {})}
    labels = {**DEFAULT_STAGE_LABELS, **metadata.get("stage_labels", {})}
    return keys, labels


def summary_policies(payload: dict) -> list[str]:
    """Every policy present in the summary, in POLICY_ORDER where known."""
    present = []
    for dataset_results in payload["datasets"].values():
        for record in dataset_results["sequential"]:
            if record["policy"] not in present:
                present.append(record["policy"])
    ranked = [p for p in POLICY_ORDER if p in present]
    return ranked + [p for p in present if p not in ranked]


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if "datasets" not in payload:
        raise ValueError(f"Expected combined cache summary with a 'datasets' field: {path}")
    return payload


def select_cache_size(payload: dict, requested_size: int | None) -> int:
    cache_sizes = payload.get("metadata", {}).get("cache_sizes", [])
    if not cache_sizes:
        raise ValueError("Combined cache summary does not include metadata.cache_sizes")
    if requested_size is None:
        return max(cache_sizes)
    if requested_size not in cache_sizes:
        raise ValueError(f"Cache size {requested_size} is not in summary cache sizes: {cache_sizes}")
    return requested_size


def find_record(records: list[dict], policy: str, cache_size: int) -> dict:
    for record in records:
        if record["policy"] == policy and record["cache_size"] == cache_size:
            return record
    raise ValueError(f"Missing record for policy={policy}, cache_size={cache_size}")


def seconds(ms: int | float) -> float:
    return ms / 1000


def dataset_breakdowns(
    dataset_results: dict, cache_size: int, policies: list[str], stage_keys: dict
) -> dict[str, dict[str, float]]:
    records = dataset_results["sequential"]
    cached_key = stage_keys["cached"]
    uncached_key = stage_keys["uncached"]
    baseline = records[0]["time_breakdown_ms"]
    breakdowns = {
        "No Cache": {
            "cached": seconds(baseline[f"{cached_key}_base"]),
            "uncached": seconds(baseline[uncached_key]),
        }
    }

    for policy in policies:
        record = find_record(records, policy, cache_size)
        timing = record["time_breakdown_ms"]
        breakdowns[POLICY_LABELS.get(policy, policy.upper())] = {
            "cached": seconds(timing[f"{cached_key}_simulated"]),
            "uncached": seconds(timing[uncached_key]),
        }

    return breakdowns


def plot_end_to_end_breakdown(payload: dict, cache_size: int, output_path: Path) -> None:
    # One dataset per vertical block instead of one per column: at a single-column
    # journal width the datasets stack, each contributing a total row and a zoomed
    # cached-stage row (rows 2*i and 2*i+1).
    datasets = list(payload["datasets"].items())
    stage_keys, stage_labels = stage_config(payload)
    policies = summary_policies(payload)
    fig, axes = plt.subplots(
        2 * len(datasets),
        1,
        # Floor the height so a single-dataset summary (RoG) still gets rows tall
        # enough for their axis labels instead of squashing into 3in.
        figsize=(FIG_WIDTH, max(DATASET_HEIGHT * len(datasets), 4.2)),
        gridspec_kw={"height_ratios": [2.2, 1.2] * len(datasets)},
        squeeze=False,
    )
    axes = axes[:, 0]
    fig.patch.set_facecolor("white")

    for i, (dataset, dataset_results) in enumerate(datasets):
        ax_total = axes[2 * i]
        ax_zoom = axes[2 * i + 1]
        ax_total.set_facecolor("white")
        ax_zoom.set_facecolor("white")

        breakdowns = dataset_breakdowns(dataset_results, cache_size, policies, stage_keys)
        labels = list(breakdowns.keys())
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        baseline_total = sum(breakdowns["No Cache"].values())
        tick_size = 9 if len(labels) <= 4 else 7

        for component in ["cached", "uncached"]:
            values = [breakdowns[label][component] for label in labels]
            ax_total.bar(
                x,
                values,
                bottom=bottom,
                color=BREAKDOWN_COLORS[component],
                edgecolor="white",
                linewidth=0.8,
                label=stage_labels[component],
                zorder=3,
            )
            bottom += values

        for i, total in enumerate(bottom):
            ax_total.text(
                x[i],
                total * 1.01,
                f"{total:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8 if len(labels) <= 4 else 6.5,
                fontweight="bold",
                color="#333333",
            )
            saved = baseline_total - total
            # `exact` saves nothing on a dataset of unique questions; printing
            # its rounding residue as "-0.0s" just looks like a bug.
            if labels[i] != "No Cache" and saved >= 0.05:
                ax_total.text(
                    x[i],
                    total * 0.94,
                    f"-{saved:.1f}s",
                    ha="center",
                    va="top",
                    fontsize=8 if len(labels) <= 4 else 6.5,
                    fontweight="bold",
                    color="#C44E52",
                )

        zoom_bottom = np.zeros(len(labels))
        for component in ["cached"]:
            values = [breakdowns[label][component] for label in labels]
            ax_zoom.bar(
                x,
                values,
                bottom=zoom_bottom,
                color=BREAKDOWN_COLORS[component],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            zoom_bottom += values

        # Inside the bar, not above it: the speedup curve on the twin axis runs
        # right through the band just above the bar tops, and with six policies
        # the two label sets collided (e.g. CWQ's "1.70x" landing on "33.8s").
        for i, value in enumerate(zoom_bottom):
            ax_zoom.text(
                x[i],
                value * 0.94,
                f"{value:.1f}s",
                ha="center",
                va="top",
                fontsize=8 if len(labels) <= 4 else 6.5,
                fontweight="bold",
                color="white",
                zorder=5,
            )

        baseline_cached = breakdowns["No Cache"]["cached"]
        speedups = [
            baseline_cached / breakdowns[label]["cached"] if breakdowns[label]["cached"] > 0 else 0.0
            for label in labels
        ]
        ax_speedup = ax_zoom.twinx()
        ax_speedup.plot(
            x,
            speedups,
            "o-",
            color=SPEEDUP_COLOR,
            linewidth=2.0,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=4,
        )
        for i, speedup in enumerate(speedups):
            ax_speedup.annotate(
                f"{speedup:.2f}x",
                xy=(x[i], speedup),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8 if len(labels) <= 4 else 6.5,
                fontweight="bold",
                color=SPEEDUP_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

        ax_total.set_title(dataset, fontsize=11, fontweight="bold", pad=8)
        ax_total.set_xticks(x)
        ax_total.set_xticklabels(labels, fontsize=tick_size, fontweight="bold")
        ax_total.tick_params(axis="x", labelbottom=True)
        ax_total.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax_total.spines[["top", "right"]].set_visible(False)

        ax_zoom.set_xticks(x)
        ax_zoom.set_xticklabels(labels, fontsize=tick_size, fontweight="bold")
        ax_zoom.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax_zoom.spines[["top", "right"]].set_visible(False)
        # Bars get the bottom ~64% of the panel and the speedup curve is mapped
        # into the top band, so the line and its labels can never cross the bars
        # or their values. Both limits are derived from the data: Belady and the
        # preload oracle reach ~2-3.6x, which the old fixed (0.75, 1.4) window
        # for WebQSP would have silently clipped off the top of the panel.
        ax_zoom.set_ylim(0, max(zoom_bottom) * 1.55)
        low, high = min(speedups), max(speedups)
        if high - low < 1e-9:
            ax_speedup.set_ylim(low - 1.0, low + 0.45)
        else:
            span = (high - low) / 0.25          # curve occupies 25% of the axis
            bottom_limit = low - 0.68 * span    # placed 68-93% up the panel
            ax_speedup.set_ylim(bottom_limit, bottom_limit + span)
            # Ticks only across the range the data actually spans. The band
            # trick above puts the axis origin far below the curve, and the
            # default locator would happily print negative "speedups" there.
            ax_speedup.set_yticks(np.linspace(low, high, 3))
        ax_speedup.spines["top"].set_visible(False)
        ax_speedup.tick_params(axis="y", colors=SPEEDUP_COLOR)
        ax_speedup.yaxis.label.set_color(SPEEDUP_COLOR)
        ax_speedup.set_ylabel(f"{stage_labels['cached']} Speedup", fontsize=11)
        ax_speedup.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1f}x"))

        # Stacked blocks have no shared left column, so every row is labelled.
        ax_total.set_ylabel("End-to-End Time (s)", fontsize=11)
        ax_zoom.set_ylabel(f"{stage_labels['cached']} Time (s)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=1.6, h_pad=1.4)
    # Anchored by its lower edge to the top of the figure, so it sits clear of
    # the first dataset title instead of landing on top of it.
    fig.legend(
        handles,
        labels,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        framealpha=0.85,
        edgecolor="#cccccc",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sequential cache time breakdowns from combined cache simulation results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-size", type=int, help="Cache size for the policy bars; defaults to largest in input.")
    parser.add_argument("--output-name", help="Override the output PDF filename.")
    args = parser.parse_args()

    payload = load_summary(args.input)
    cache_size = select_cache_size(payload, args.cache_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    name = args.output_name or f"cache_time_breakdown_e2e_size_{cache_size}.pdf"
    plot_end_to_end_breakdown(payload, cache_size, args.output_dir / name)


if __name__ == "__main__":
    main()
