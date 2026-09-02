#!/usr/bin/env python3
"""Accuracy-vs-cache-size bar chart for the Gemini runs (ToG and RoG, WebQSP and CWQ).

Companion to accuracy_plot.py (which stays a per-policy bar chart). This one
sweeps the cache *capacity* instead:

    size 0        -> the uncached baseline, policy "none"
    size 128/512/4096 -> policy "semantic_lfu"

4096 is plotted as "inf": the whole workload (~1.6k questions) fits under that
capacity, so nothing is ever evicted and it is an effectively unbounded cache.

One cluster per (dataset, system), so the chart shows how much end-to-end answer
quality moves as the semantic cache grows, and whether that holds on the harder
multi-hop split.

Sources -- gemini, and the live end-to-end runs on both datasets (--pipeline live,
the default):
  WebQSP  RoG  artifacts/rog_cache/rog_live_virt_gemini[_<size>]/summary.json
          ToG  compare_results/tog_rerun_live_virt_gemini/summary.json at size
               0 and 4096, compare_results/tog_live_virt_gemini_<size> at 128
               and 512 -- ToG spells the unsized run with `rerun_` and the sized
               ones without it, same configuration either way
  CWQ     RoG  artifacts/rog_cache/rog_cwq_live_virt_gemini[_<size>]/summary.json
          ToG  compare_results/tog_cwq_live_virt_gemini[_<size>]/summary.json

`--pipeline replay` swaps WebQSP for the replayed sweep instead
(gemini_<sys>_cache_<backend>_test[_<size>], ToG preferring the run's first cold
pass), which is what this chart plotted before. It covers WebQSP only: the replay
family has no `none` for ToG on CWQ (compare_results/gemini_tog_cache_virtuoso_cwq
holds semantic_lfu and semantic_lru only), so there is no uncached run to anchor
size 0 against, and asking for both is refused rather than quietly mixing the two.

One pipeline across both datasets is the point of the default: the WebQSP and CWQ
clusters then differ in the dataset and the question count (1628/1639 against 400)
rather than in the dataset, the count AND how the runs were produced. The
per-point provenance printed underneath still names the run and its n for each
bar, because 400 questions is a wide interval to read a 1-point difference in.

CWQ has no 512 run. That bar is left out rather than filled in from a
neighbouring capacity; the provenance lists it as `-- not run`.

    python src/ToG-cache/output/accuracy_vs_size_plot.py
    python src/ToG-cache/output/accuracy_vs_size_plot.py --datasets webqsp
    python src/ToG-cache/output/accuracy_vs_size_plot.py --metric hit
    python src/ToG-cache/output/accuracy_vs_size_plot.py \
        --pipeline replay --datasets webqsp --backend oxi
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

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
DATASET_LABELS = {"webqsp": "WebQSP", "cwq": "CWQ"}
# Same palette/hatch order as accuracy_plot.py, so the two charts read as a pair.
COLORS = ['#999999', '#56B4E9', '#E69F00', '#009E73']
HATCHES = ['', '/', '\\', '++']


def run_tag(system: str, backend: str, dataset: str, size: int,
            pipeline: str = "live") -> str:
    """Run directory name for a system/backend/dataset/capacity."""
    if dataset == "cwq":
        # Only ever run live, and only against Virtuoso -- see the note at the
        # top of this file for why the replay family cannot stand in.
        base = sized = f"{system}_cwq_live_virt_gemini"
    elif pipeline == "live":
        # ToG's unsized live run is on disk as `tog_rerun_live_virt_gemini` while
        # its capacity sweeps are `tog_live_virt_gemini_<size>`; RoG uses the one
        # spelling for both. Same configuration either way -- scripts/
        # average_replicates.py checks the pair against run_config.json.
        base = "tog_rerun_live_virt_gemini" if system == "tog" else "rog_live_virt_gemini"
        sized = f"{system}_live_virt_gemini"
    else:
        base = sized = f"gemini_{system}_cache_{backend}_test"
    return base if size in (0, DEFAULT_CAPACITY) else f"{sized}_{size}"


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


def load_point(system: str, backend: str, dataset: str, size: int, metric: str,
               pipeline: str = "live") -> tuple[float, int, Path] | None:
    """(percent, n_questions, source path) for one bar, or None if it was never run."""
    tag = run_tag(system, backend, dataset, size, pipeline)
    policy = BASELINE_POLICY if size == 0 else CACHED_POLICY

    candidates: list[tuple[Path, callable]] = []
    if system == "tog":
        # Prefer the 1st (cold) pass of a loop run, matching accuracy_plot.py.
        candidates.append((ART["tog"] / f"{tag}_pass1" / "summary.json", read_artifact_summary))
        # Then the rescored summary, ahead of the run's own. ToG's stock extractor
        # (eval/utils.py clean_results) scans to the FIRST braced span, which on a
        # live run is the `{Yes}` sufficiency marker rather than the answer -- worth
        # ~29 Hits@1 points on tog_rerun_live_virt_gemini (36.7 stored, 66.3
        # rescored). scripts/rescore_tog.py writes the corrected summary; the loop
        # runs above are unaffected (they move by ~0.3) so their cold pass still wins.
        candidates.append((COMPARE_RESULTS / f"{tag}_rescored" / "summary.json",
                           read_compare_summary))
        candidates.append((COMPARE_RESULTS / tag / "summary.json", read_compare_summary))
    candidates.append((ART[system] / tag / "summary.json", read_artifact_summary))

    for path, reader in candidates:
        if not path.exists():
            continue
        recs = reader(path, metric)
        if policy in recs:
            value, n = recs[policy]
            return value, n, path
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="virtuoso", choices=["virtuoso", "oxi"],
                    help="SPARQL backend the WebQSP runs used (default: virtuoso). "
                         "CWQ was only run against Virtuoso.")
    ap.add_argument("--metric", default="accuracy", choices=list(METRIC_LABELS),
                    help="which measured metric to plot (default: accuracy)")
    ap.add_argument("--sizes", default="0,128,512,4096",
                    help="comma-separated cache capacities on the x axis")
    ap.add_argument("--systems", default="tog,rog",
                    help="comma-separated systems to plot (tog, rog)")
    ap.add_argument("--datasets", default="webqsp,cwq",
                    help="comma-separated datasets to plot (webqsp, cwq)")
    ap.add_argument("--pipeline", default="live", choices=["live", "replay"],
                    help="which WebQSP runs to plot (default: live, the same "
                         "pipeline CWQ is measured on). `replay` is the replayed "
                         "sweep this chart used to plot, and covers WebQSP only.")
    ap.add_argument("-o", "--output",
                    default=str(Path(__file__).resolve().parent / "cache_vs_accuracy_gemini.pdf"),
                    help="output PDF path")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for dataset in datasets:
        if dataset not in DATASET_LABELS:
            raise SystemExit(f"unknown dataset {dataset!r}; "
                             f"choose from {', '.join(DATASET_LABELS)}")
    if "cwq" in datasets and args.pipeline != "live":
        raise SystemExit("CWQ has no replayed sweep with an uncached baseline; "
                         "use --pipeline live or drop --datasets cwq")
    if args.backend != "virtuoso" and (args.pipeline == "live" or "cwq" in datasets):
        raise SystemExit("the live sweeps were only run against the virtuoso "
                         "backend; --backend oxi needs --pipeline replay "
                         "--datasets webqsp")

    # One cluster per (dataset, system), datasets kept together on the axis.
    clusters = [(dataset, system) for dataset in datasets for system in systems]

    series: dict[tuple[str, str], list[float]] = {}
    provenance: list[str] = []
    for dataset, system in clusters:
        values = []
        for size in sizes:
            policy = BASELINE_POLICY if size == 0 else CACHED_POLICY
            point = load_point(system, args.backend, dataset, size, args.metric,
                               args.pipeline)
            if point is None:
                # A capacity that was never swept on this dataset. Left as a hole
                # in the cluster rather than borrowed from another capacity.
                values.append(math.nan)
                provenance.append(f"  {dataset:>6} {system:>3} size={size:>5} "
                                  f"{policy:<13}     --  not run")
                continue
            value, n, path = point
            values.append(value / 100.0)
            mark = " [rescored]" if path.parent.name.endswith("_rescored") else ""
            provenance.append(f"  {dataset:>6} {system:>3} size={size:>5} "
                              f"{policy:<13} {value:6.2f}  n={n:<5} "
                              f"{path.relative_to(REPO_ROOT)}{mark}")
        if all(math.isnan(v) for v in values):
            raise SystemExit(f"no runs found for {system} on {dataset} at any of "
                             f"{args.sizes}")
        series[(dataset, system)] = values

    # Journal (single-column body) style: the figure spans the ~6.5in text block
    # and is printed at 100%, so fonts match body text instead of being inflated
    # for an IEEE two-column shrink.
    fig_w = 6.0 if len(clusters) <= 2 else 6.9
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'figure.figsize': (fig_w, 3.0),
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(fig_w, 3.0))
    # One cluster per (dataset, system), one bar per cache size inside it, with
    # a wider gap between datasets than between the systems within one.
    group_pos: list[float] = []
    x = 0.0
    for i, (dataset, _system) in enumerate(clusters):
        if i and dataset != clusters[i - 1][0]:
            x += 0.55
        group_pos.append(x)
        x += 1.15
    group_pos = np.array(group_pos)
    width = 0.68 / len(sizes)

    for i, size in enumerate(sizes):
        offset = (i - (len(sizes) - 1) / 2) * width
        values = [series[c][i] for c in clusters]
        bars = ax.bar(group_pos + offset, values, width=width * 0.92,
                      color=COLORS[i % len(COLORS)], edgecolor='#555555',
                      linewidth=0.6,
                      hatch=HATCHES[i % len(HATCHES)],
                      label=SIZE_LABELS.get(size, str(size)))
        for bar, v in zip(bars, values):
            if math.isnan(v):
                # Say so in the gap: an empty slot alone reads as a zero score.
                ax.annotate('n/a', (bar.get_x() + bar.get_width() / 2, 0),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6.5,
                            color='#999999', rotation=90)
                continue
            # Value labels on top -- accuracy spreads across sizes are small.
            ax.annotate(f"{100 * v:.1f}", (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5, rotation=90)

    ax.set_ylabel(f'{METRIC_LABELS[args.metric]} (%)')
    ax.set_xticks(group_pos)
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s.upper()) for _d, s in clusters],
                       fontweight='bold')
    if len(datasets) > 1:
        # Dataset named once under the block of systems it covers, so the x axis
        # carries both levels without repeating "ToG (WebQSP)" on every tick.
        ax.set_xlabel('')
        span = blended_transform_factory(ax.transData, ax.transAxes)
        for dataset in datasets:
            xs = [p for p, (d, _s) in zip(group_pos, clusters) if d == dataset]
            ax.text(sum(xs) / len(xs), -0.20, DATASET_LABELS[dataset],
                    transform=span, ha='center', va='top', fontsize=9)
        ax.tick_params(axis='x', length=0)
    else:
        ax.set_xlabel('System')
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{100 * value:.0f}'))

    max_v = max(v for vals in series.values() for v in vals if not math.isnan(v))
    ax.set_ylim(0, max_v * 1.12)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    # Legend sits above the axes so it never collides with the value labels.
    ax.legend(title='Cache Size', frameon=False, ncol=len(sizes),
              loc='lower center', bbox_to_anchor=(0.5, 1.02),
              fontsize=9, title_fontsize=9,
              columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    plt.tight_layout()

    plt.savefig(args.output, format='pdf')
    print(f"wrote {args.output}  [gemini, {args.backend}, {args.metric}, "
          f"{'+'.join(datasets)}, {args.pipeline}]")
    print("\n".join(provenance))


if __name__ == "__main__":
    main()
