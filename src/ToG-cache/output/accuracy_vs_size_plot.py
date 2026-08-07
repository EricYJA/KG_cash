#!/usr/bin/env python3
"""Accuracy-vs-cache-size line chart for the Gemini runs (ToG and RoG).

Companion to accuracy_plot.py (which stays a per-policy bar chart). This one
sweeps the cache *capacity* instead:

    size 0        -> the uncached baseline, policy "none"
    size 128/512/4096 -> policy "semantic_lfu"

4096 is plotted as "inf": the whole workload (~1.6k questions) fits under that
capacity, so nothing is ever evicted and it is an effectively unbounded cache.

Both systems are plotted as separate series, so the chart shows how much the
end-to-end answer quality moves as the semantic cache grows.

Sources (gemini, virtuoso backend by default):
  RoG  artifacts/rog_cache/gemini_rog_cache_<backend>_test[_<size>]/summary.json
  ToG  artifacts/tog_cache/gemini_tog_cache_<backend>_test_pass1/summary.json
       (first cold pass; the 128/512 sweeps have no loop runs, so those fall
        back to src/ToG-cache/output/compare_results/<run>_<size>/summary.json)

    python src/ToG-cache/output/accuracy_vs_size_plot.py
    python src/ToG-cache/output/accuracy_vs_size_plot.py --metric hit --backend oxi
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
COMPARE_RESULTS = Path(__file__).resolve().parent / "compare_results"

DEFAULT_CAPACITY = 4096          # runs without a size suffix were run at this capacity
# Nothing is ever evicted at the default capacity -- label it as unbounded.
SIZE_LABELS = {DEFAULT_CAPACITY: r"$\infty$"}
BASELINE_POLICY = "none"         # what size 0 means
CACHED_POLICY = "semantic_lfu"   # what every non-zero size means

METRIC_LABELS = {"accuracy": "Accuracy", "hit": "Hits@1", "f1": "F1"}
# compare_results/*/summary.json uses different key names, and 0-1 fractions.
COMPARE_KEYS = {"accuracy": "recall", "hit": "hits1", "f1": "f1"}

SYSTEM_LABELS = {"tog": "ToG", "rog": "RoG"}
# Same palette/hatch order as accuracy_plot.py, so the two charts read as a pair.
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
HATCHES = ['/', '\\', '|', '-', 'x']


def run_tag(system: str, backend: str, size: int) -> str:
    """Run directory name for a system/backend/capacity."""
    base = f"gemini_{system}_cache_{backend}_test"
    return base if size in (0, DEFAULT_CAPACITY) else f"{base}_{size}"


def read_artifact_summary(path: Path, metric: str) -> dict[str, tuple[float, int]]:
    """{policy: (percent, n_questions)} from an artifacts/*/summary.json record list."""
    records = json.loads(path.read_text())
    return {
        r["policy"]: (float(r.get(metric) or 0.0), int(r.get("n_questions") or 0))
        for r in records
    }


def read_compare_summary(path: Path, metric: str) -> dict[str, tuple[float, int]]:
    """Same shape, from a compare_results/*/summary.json {args, rows} blob (fractions)."""
    rows = json.loads(path.read_text())["rows"]
    key = COMPARE_KEYS[metric]
    return {
        r["policy"]: (100.0 * float(r.get(key) or 0.0), int(r.get("records") or 0))
        for r in rows
    }


def load_point(system: str, backend: str, size: int, metric: str) -> tuple[float, int, Path]:
    """(percent, n_questions, source path) for one (system, size) point."""
    tag = run_tag(system, backend, size)
    policy = BASELINE_POLICY if size == 0 else CACHED_POLICY

    candidates: list[tuple[Path, callable]] = []
    if system == "tog":
        # Prefer the 1st (cold) pass of a loop run, matching accuracy_plot.py.
        candidates.append((ART["tog"] / f"{tag}_pass1" / "summary.json", read_artifact_summary))
        candidates.append((COMPARE_RESULTS / tag / "summary.json", read_compare_summary))
    candidates.append((ART[system] / tag / "summary.json", read_artifact_summary))

    for path, reader in candidates:
        if not path.exists():
            continue
        recs = reader(path, metric)
        if policy in recs:
            value, n = recs[policy]
            return value, n, path
    looked = ", ".join(str(p) for p, _ in candidates)
    raise SystemExit(f"no {policy!r} record for {system} size {size} (looked for {looked})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="virtuoso", choices=["virtuoso", "oxi"],
                    help="SPARQL backend the runs used (default: virtuoso)")
    ap.add_argument("--metric", default="accuracy", choices=list(METRIC_LABELS),
                    help="which measured metric to plot (default: accuracy)")
    ap.add_argument("--sizes", default="0,128,512,4096",
                    help="comma-separated cache capacities on the x axis")
    ap.add_argument("--systems", default="tog,rog",
                    help="comma-separated systems to plot (tog, rog)")
    ap.add_argument("-o", "--output",
                    default=str(Path(__file__).resolve().parent / "cache_vs_accuracy_gemini.pdf"),
                    help="output PDF path")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    series: dict[str, list[float]] = {}
    provenance: list[str] = []
    for system in systems:
        values = []
        for size in sizes:
            value, n, path = load_point(system, args.backend, size, args.metric)
            values.append(value / 100.0)
            policy = BASELINE_POLICY if size == 0 else CACHED_POLICY
            provenance.append(f"  {system:>3} size={size:>5} {policy:<13} "
                              f"{value:6.2f}  n={n:<5} {path.relative_to(REPO_ROOT)}")
        series[system] = values

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
    # One cluster per system, one bar per cache size inside it.
    group_pos = np.arange(len(systems))
    width = 0.8 / len(sizes)

    for i, size in enumerate(sizes):
        offset = (i - (len(sizes) - 1) / 2) * width
        values = [series[system][i] for system in systems]
        bars = ax.bar(group_pos + offset, values, width=width * 0.92,
                      color=COLORS[i % len(COLORS)], edgecolor='black',
                      hatch=HATCHES[i % len(HATCHES)],
                      label=SIZE_LABELS.get(size, str(size)))
        for bar, v in zip(bars, values):
            # Value labels on top -- accuracy spreads across sizes are small.
            ax.annotate(f"{100 * v:.1f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('System')
    ax.set_ylabel(METRIC_LABELS[args.metric])
    ax.set_xticks(group_pos)
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s.upper()) for s in systems])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    max_v = max(v for vals in series.values() for v in vals)
    ax.set_ylim(0, max_v * 1.22)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    # Legend sits above the axes so it never collides with the value labels.
    ax.legend(title='Cache Size', frameon=False, ncol=len(sizes),
              loc='lower center', bbox_to_anchor=(0.5, 1.02),
              fontsize=9, title_fontsize=9,
              columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    plt.tight_layout()

    plt.savefig(args.output, format='pdf', bbox_inches='tight')
    print(f"wrote {args.output}  [gemini, {args.backend}, {args.metric}]")
    print("\n".join(provenance))


if __name__ == "__main__":
    main()
