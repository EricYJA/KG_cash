#!/usr/bin/env python3
"""Accuracy-by-cache-policy bar chart, from the LIVE run results.

This used to hardcode an accuracy-vs-cache-size series. The live experiments run
one cache *capacity* per run (there is no accuracy-vs-capacity sweep to read), but
they do measure end-to-end accuracy for every cache *policy*. So the chart now
reads a real run summary under artifacts/{tog,rog}_cache/ and plots first-pass
accuracy (or Hits@1 / F1) per policy, with None as the uncached baseline.

    PYTHONPATH=src python src/ToG-cache/output/accuracy_plot.py --run gemini_tog_cache_oxi_test
    python src/ToG-cache/output/accuracy_plot.py --run rog_cache_virtuoso_test --metric hit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[3]
ART = {
    "tog": REPO_ROOT / "artifacts" / "tog_cache",
    "rog": REPO_ROOT / "artifacts" / "rog_cache",
}

POLICY_ORDER = ["none", "exact", "semantic_lru", "semantic_lfu", "semantic_oracle"]
POLICY_LABELS = {
    "none": "None", "exact": "Exact", "semantic_lru": "Sem-LRU",
    "semantic_lfu": "Sem-LFU", "semantic_oracle": "Sem-Oracle",
}
METRIC_LABELS = {"accuracy": "Accuracy", "hit": "Hits@1", "f1": "F1"}


def load_summary(run: str, first_pass: bool) -> tuple[dict[str, dict], Path]:
    """{policy: record} for a run; prefers the 1st-pass (cold) ToG summary."""
    system = "tog" if "tog" in run.lower() else "rog"
    path = ART[system] / run / "summary.json"
    if first_pass:
        p1 = ART["tog"] / f"{run}_pass1" / "summary.json"   # ToG loop runs only
        if p1.exists():
            path = p1
    if not path.exists():
        raise SystemExit(f"no summary for run {run!r} (looked for {path})")
    records = json.loads(path.read_text())
    return {r["policy"]: r for r in records}, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="gemini_tog_cache_oxi_test",
                    help="run tag under artifacts/{tog,rog}_cache/")
    ap.add_argument("--metric", default="accuracy", choices=list(METRIC_LABELS),
                    help="which measured metric to plot (default: accuracy)")
    ap.add_argument("--whole", action="store_true",
                    help="use the whole-run summary instead of the 1st pass")
    ap.add_argument("-o", "--output",
                    default=str(Path(__file__).resolve().parent / "cache_vs_accuracy.pdf"),
                    help="output PDF path")
    args = ap.parse_args()

    recs, path = load_summary(args.run, first_pass=not args.whole)
    policies = [p for p in POLICY_ORDER if p in recs]
    labels = [POLICY_LABELS.get(p, p) for p in policies]
    # Summary metrics are 0-100 percentages; scale to 0-1 for PercentFormatter.
    values = [(recs[p].get(args.metric) or 0.0) / 100.0 for p in policies]

    # Journal (single-column body) style: the figure spans the ~6.5in text block
    # and is printed at 100%, so fonts match body text instead of being inflated
    # for an IEEE two-column shrink.
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6.5, 3.0),
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots()
    x_pos = np.arange(len(policies))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    hatch_patterns = ['/', '\\', '|', '-', 'x']

    bars = ax.bar(x_pos, values, color=[colors[i % len(colors)] for i in x_pos],
                  edgecolor='black', width=0.6)
    for i, bar in enumerate(bars):
        bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])
        # Value label on top -- accuracy spreads across policies are small.
        ax.annotate(f"{100 * values[i]:.1f}", (bar.get_x() + bar.get_width() / 2,
                    values[i]), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Cache Policy')
    ax.set_ylabel(METRIC_LABELS[args.metric])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    max_v = max(values) if values else 0
    ax.set_ylim(0, (max_v * 1.18) if max_v > 0 else 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    plt.savefig(args.output, format='pdf', bbox_inches='tight')
    pass_note = "whole-run" if args.whole else "1st pass"
    print(f"wrote {args.output}  [{args.run}, {args.metric}, {pass_note}] from {path}")


if __name__ == "__main__":
    main()
