from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "output" / "cache_sim_summary.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
POLICIES = ["lru", "lfu"]
POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
}
BREAKDOWN_COLORS = {
    "kg": "#4C72B0",
    "llm": "#55A868",
}
SPEEDUP_COLOR = "#C44E52"


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


def dataset_breakdowns(dataset_results: dict, cache_size: int) -> dict[str, dict[str, float]]:
    records = dataset_results["sequential"]
    baseline = records[0]["time_breakdown_ms"]
    breakdowns = {
        "No Cache": {
            "kg": seconds(baseline["kg_base"]),
            "llm": seconds(baseline["llm"]),
        }
    }

    for policy in POLICIES:
        record = find_record(records, policy, cache_size)
        timing = record["time_breakdown_ms"]
        breakdowns[POLICY_LABELS[policy]] = {
            "kg": seconds(timing["kg_simulated"]),
            "llm": seconds(timing["llm"]),
        }

    return breakdowns


def plot_end_to_end_breakdown(payload: dict, cache_size: int, output_path: Path) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 6.6),
        sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )
    fig.patch.set_facecolor("white")

    for col, (dataset, dataset_results) in enumerate(payload["datasets"].items()):
        ax_total = axes[0, col]
        ax_zoom = axes[1, col]
        ax_total.set_facecolor("white")
        ax_zoom.set_facecolor("white")

        breakdowns = dataset_breakdowns(dataset_results, cache_size)
        labels = list(breakdowns.keys())
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        baseline_total = sum(breakdowns["No Cache"].values())

        for component in ["kg", "llm"]:
            values = [breakdowns[label][component] for label in labels]
            ax_total.bar(
                x,
                values,
                bottom=bottom,
                color=BREAKDOWN_COLORS[component],
                edgecolor="white",
                linewidth=0.8,
                label=component.upper() if component != "other" else "Other",
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
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )
            if labels[i] != "No Cache":
                saved = baseline_total - total
                ax_total.text(
                    x[i],
                    total * 0.94,
                    f"-{saved:.1f}s",
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    color="#C44E52",
                )

        zoom_bottom = np.zeros(len(labels))
        for component in ["kg"]:
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

        for i, value in enumerate(zoom_bottom):
            ax_zoom.text(
                x[i],
                value * 1.03,
                f"{value:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#333333",
            )

        baseline_kg = breakdowns["No Cache"]["kg"]
        speedups = [
            baseline_kg / breakdowns[label]["kg"] if breakdowns[label]["kg"] > 0 else 0.0
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
                xytext=(0, 16),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=SPEEDUP_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

        ax_total.set_title(dataset, fontsize=13, fontweight="bold", pad=12)
        ax_total.set_xticks(x)
        ax_total.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax_total.tick_params(axis="x", labelbottom=True)
        ax_total.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax_total.spines[["top", "right"]].set_visible(False)

        ax_zoom.set_xticks(x)
        ax_zoom.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax_zoom.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax_zoom.spines[["top", "right"]].set_visible(False)
        ax_zoom.set_ylim(0, max(zoom_bottom) * 1.22)
        if dataset == "WebQSP":
            ax_speedup.set_ylim(0.75, 1.4)
        else:
            ax_speedup.set_ylim(0.75, max(speedups) * 1.28)
        ax_speedup.spines["top"].set_visible(False)
        ax_speedup.tick_params(axis="y", colors=SPEEDUP_COLOR)
        ax_speedup.yaxis.label.set_color(SPEEDUP_COLOR)
        ax_speedup.set_ylabel("KG Speedup", fontsize=11)
        ax_speedup.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1f}x"))

    axes[0, 0].set_ylabel("End-to-End Time (s)", fontsize=11)
    axes[1, 0].set_ylabel("KG Time (s)", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        framealpha=0.85,
        edgecolor="#cccccc",
    )
    fig.tight_layout(pad=2.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sequential cache time breakdowns from combined cache simulation results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-size", type=int, help="Cache size for LRU/LFU bars; defaults to largest in input.")
    args = parser.parse_args()

    payload = load_summary(args.input)
    cache_size = select_cache_size(payload, args.cache_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_end_to_end_breakdown(
        payload,
        cache_size,
        args.output_dir / f"cache_time_breakdown_e2e_size_{cache_size}.pdf",
    )


if __name__ == "__main__":
    main()
