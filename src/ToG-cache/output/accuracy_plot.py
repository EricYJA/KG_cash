#!/usr/bin/env python3
"""Accuracy-by-cache-policy bar chart, from the LIVE run results.

This used to hardcode an accuracy-vs-cache-size series. The live experiments run
one cache *capacity* per run (there is no accuracy-vs-capacity sweep to read), but
they do measure end-to-end accuracy for every cache *policy*. So the chart now
reads a real run summary under artifacts/{tog,rog}_cache/ and plots first-pass
accuracy (or Hits@1 / F1) per policy, with None as the uncached baseline.

Both datasets sit in one policy group where the run has a CWQ counterpart, which
is what CWQ_COUNTERPART lists: policy still drives the colour and hatch, so the
chart reads per policy exactly as it did, and the dataset is the shade -- WebQSP
at full strength, CWQ tinted, named once in the legend. A run with no counterpart
(or --no-cwq) draws the single series it always drew.

CWQ has no `exact` run on either system. That bar is left out rather than filled
in from the WebQSP one, and is marked `n/a` in its slot.

    PYTHONPATH=src python src/ToG-cache/output/accuracy_plot.py
    python src/ToG-cache/output/accuracy_plot.py --run rog_live_virt_gemini
    python src/ToG-cache/output/accuracy_plot.py --run gemini_tog_cache_oxi_test --metric hit
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
COMPARE_DIR = Path(__file__).resolve().parent / "compare_results"

POLICY_ORDER = ["none", "exact", "semantic_lru", "semantic_lfu", "semantic_oracle"]
POLICY_LABELS = {
    "none": "None", "exact": "Exact", "semantic_lru": "Sem-LRU",
    "semantic_lfu": "Sem-LFU", "semantic_oracle": "Sem-Oracle",
}
METRIC_LABELS = {"accuracy": "Accuracy", "hit": "Hits@1", "f1": "F1"}

# The live WebQSP runs that have a CWQ run of the same configuration. Only these
# pairs exist with an uncached baseline on both sides; every other tag draws the
# single WebQSP series.
CWQ_COUNTERPART = {
    "tog_rerun_live_virt_gemini": "tog_cwq_live_virt_gemini",
    "rog_live_virt_gemini": "rog_cwq_live_virt_gemini",
}
DATASET_LABELS = {"webqsp": "WebQSP", "cwq": "CWQ"}
# How far the CWQ bars are blended toward white: enough to read as the second
# member of a pair, not so far that the policy hue stops being the policy hue.
CWQ_TINT = 0.45


def read_compare_summary(path: Path) -> dict[str, dict]:
    """compare_results/*/summary.json (0-1 fractions) in the artifacts record shape."""
    return {
        r["policy"]: {
            "policy": r["policy"],
            "hit": 100.0 * (r.get("hits1") or 0.0),
            "accuracy": 100.0 * (r.get("recall") or 0.0),
            "recall": 100.0 * (r.get("recall") or 0.0),
            "precision": 100.0 * (r.get("precision") or 0.0),
            "f1": 100.0 * (r.get("f1") or 0.0),
        }
        for r in json.loads(path.read_text())["rows"]
    }


def load_summary(run: str, first_pass: bool) -> tuple[dict[str, dict], Path]:
    """{policy: record} for a run; prefers the 1st-pass (cold) ToG summary.

    Then, for ToG, the rescored summary ahead of the run's own. ToG's stock
    extractor (eval/utils.py clean_results) scans to the FIRST braced span, which
    on a live run is the `{Yes}` sufficiency marker rather than the answer -- worth
    ~29 Hits@1 points on tog_rerun_live_virt_gemini (36.7 stored, 66.3 rescored).
    scripts/rescore_tog.py writes the corrected summary; the loop runs move by
    ~0.3 under it, so their cold pass still wins where one exists.
    """
    system = "tog" if "tog" in run.lower() else "rog"
    if first_pass:
        p1 = ART["tog"] / f"{run}_pass1" / "summary.json"   # ToG loop runs only
        if p1.exists():
            return {r["policy"]: r for r in json.loads(p1.read_text())}, p1
    if system == "tog":
        resc = COMPARE_DIR / f"{run}_rescored" / "summary.json"
        if resc.exists():
            return read_compare_summary(resc), resc
    path = ART[system] / run / "summary.json"
    if not path.exists():
        raise SystemExit(f"no summary for run {run!r} (looked for {path})")
    records = json.loads(path.read_text())
    return {r["policy"]: r for r in records}, path


def _tint(hex_color: str, amount: float) -> str:
    """Blend `hex_color` `amount` of the way toward white (0 = unchanged, 1 = white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(c + (255 - c) * amount)) for c in (r, g, b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="tog_rerun_live_virt_gemini",
                    help="WebQSP run tag under artifacts/{tog,rog}_cache/")
    ap.add_argument("--cwq-run", default="",
                    help="CWQ run to plot beside it (default: the counterpart of "
                         "--run, where one exists)")
    ap.add_argument("--no-cwq", action="store_true",
                    help="plot the WebQSP run alone, as this chart did before")
    ap.add_argument("--metric", default="accuracy", choices=list(METRIC_LABELS),
                    help="which measured metric to plot (default: accuracy)")
    ap.add_argument("--whole", action="store_true",
                    help="use the whole-run summary instead of the 1st pass")
    ap.add_argument("-o", "--output",
                    default=str(Path(__file__).resolve().parent / "cache_vs_accuracy.pdf"),
                    help="output PDF path")
    args = ap.parse_args()

    recs, path = load_summary(args.run, first_pass=not args.whole)
    cwq_tag = "" if args.no_cwq else (args.cwq_run
                                      or CWQ_COUNTERPART.get(args.run, ""))
    cwq_recs, cwq_path = ({}, None)
    if cwq_tag:
        cwq_recs, cwq_path = load_summary(cwq_tag, first_pass=not args.whole)

    # Union, so a policy only one dataset ran still gets its group and its label.
    policies = [p for p in POLICY_ORDER if p in recs or p in cwq_recs]
    labels = [POLICY_LABELS.get(p, p) for p in policies]

    def series(source: dict) -> list[float]:
        # Summary metrics are 0-100 percentages; scale to 0-1 for PercentFormatter.
        # A policy the run never had is nan -- a hole, not a zero score.
        return [(source[p].get(args.metric) or 0.0) / 100.0 if p in source
                else np.nan for p in policies]

    values = series(recs)
    cwq_values = series(cwq_recs) if cwq_tag else []

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

    # One series keeps the full-width bar this chart has always drawn; two split
    # that slot, so the policy groups sit where they did and only the bars narrow.
    single = not cwq_tag
    width = 0.6 if single else 0.34
    # A hairline between the pair, so two shades of one hue read as two bars.
    gap = width / 2 + 0.012
    drawn = [(values, 0.0, 0.0)] if single else [(values, -gap, 0.0),
                                                 (cwq_values, gap, CWQ_TINT)]
    for vals, offset, tint in drawn:
        bars = ax.bar(x_pos + offset, vals,
                      color=[_tint(colors[i % len(colors)], tint) for i in x_pos],
                      edgecolor='black', width=width)
        for i, bar in enumerate(bars):
            bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])
            xc = bar.get_x() + bar.get_width() / 2
            if np.isnan(vals[i]):
                # Say so in the gap: an empty slot alone reads as a zero score.
                ax.annotate('n/a', (xc, 0.0), xytext=(0, 3),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=6.5, color='#999999', rotation=90)
                continue
            # Value label on top -- accuracy spreads across policies are small.
            ax.annotate(f"{100 * vals[i]:.1f}", (xc, vals[i]), ha='center',
                        va='bottom', fontsize=8, rotation=0 if single else 90,
                        xytext=(0, 2), textcoords='offset points')

    ax.set_xlabel('Cache Policy')
    ax.set_ylabel(METRIC_LABELS[args.metric])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    finite = [v for v in [*values, *cwq_values] if not np.isnan(v)]
    max_v = max(finite) if finite else 0
    # Rotated value labels stand taller than flat ones, so the paired chart needs
    # a little more headroom above the tallest bar than the single one.
    ax.set_ylim(0, (max_v * (1.18 if single else 1.30)) if max_v > 0 else 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    if not single:
        # Neutral swatches: the colours already mean policy, so a per-dataset
        # legend keyed to them would claim a meaning they do not carry.
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='#8c8c8c', edgecolor='black',
                                 label=DATASET_LABELS['webqsp']),
                           Patch(facecolor=_tint('#8c8c8c', CWQ_TINT),
                                 edgecolor='black', label=DATASET_LABELS['cwq'])],
                  frameon=False, ncol=2, loc='upper right',
                  handlelength=1.4, handletextpad=0.5, columnspacing=1.0)
    plt.tight_layout()

    plt.savefig(args.output, format='pdf', bbox_inches='tight')
    pass_note = "whole-run" if args.whole else "1st pass"
    print(f"wrote {args.output}  [{args.run}, {args.metric}, {pass_note}] from {path}")
    if cwq_tag:
        print(f"  + CWQ [{cwq_tag}] from {cwq_path}")
        missing = [POLICY_LABELS.get(p, p) for p in policies if p not in cwq_recs]
        if missing:
            print(f"  CWQ has no {', '.join(missing)} run; left as n/a")


if __name__ == "__main__":
    main()
