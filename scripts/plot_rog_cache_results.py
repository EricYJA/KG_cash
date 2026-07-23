#!/usr/bin/env python3
"""Plot RoG cache-experiment results from one or more summary.json files.

Each run's summary.json (written by summarize_rog_cache.py) is a list of per-policy
records with: policy, hit (Hits@1), f1, accuracy, hit_rate, speedup_x,
full_speedup_x. Policies go on the x-axis and each metric gets its own panel
(small multiples -- never a dual y-axis). One line per run, styled to match the
characterization/ figures (IEEE serif, tab10 colours, shared bottom legend, PDF).

    # one run
    python scripts/plot_rog_cache_results.py --runs rog_cache_virtuoso_test

    # compare runs (one line per run)
    python scripts/plot_rog_cache_results.py \
        --runs rog_cache_virtuoso_test gemini_rog_cache_virtuoso_test

    # all runs found under artifacts/rog_cache/
    python scripts/plot_rog_cache_results.py --all

A --runs value may be a tag (resolved to artifacts/rog_cache/<tag>/summary.json),
a directory, or a direct path to a summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROG_DIR = REPO_ROOT / "artifacts" / "rog_cache"

# Fixed policy order + short labels (baseline first, then caching policies).
POLICY_ORDER = ["none", "exact", "semantic_lfu", "semantic_lru", "semantic_oracle"]
POLICY_LABELS = {
    "none": "None",
    "exact": "Exact",
    "semantic_lfu": "Sem-LFU",
    "semantic_lru": "Sem-LRU",
    "semantic_oracle": "Sem-Oracle",
}

# characterization/ house style: tab10 colours + grayscale-safe markers/linestyles.
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "--", "-.", ":", "-"]

ACC_TITLES = {"hit": "Hits@1 (%)", "accuracy": "Accuracy (%)", "f1": "F1 (%)"}


def pretty_label(tag: str) -> str:
    """Human-readable legend label derived from a run tag: model + KG backend.

    gemini* -> "Gemini 3.1 flash-lite", otherwise "Haiku 4.5"; *virtuoso* adds
    "(Virtuoso)", *oxi* adds "(Oxigraph)".
    """
    t = tag.lower()
    model = "Gemini 3.1 flash-lite" if "gemini" in t else "Haiku 4.5"
    if "virtuoso" in t:
        return f"{model} (Virtuoso)"
    if "oxi" in t:
        return f"{model} (Oxigraph)"
    return model


def resolve_summary(run: str) -> Path:
    """Accept a tag, a dir, or a direct summary.json path."""
    p = Path(run)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "summary.json"
    return ROG_DIR / run / "summary.json"


def load_run(path: Path) -> dict[str, dict]:
    """Return {policy: record} for one run's summary.json."""
    records = json.loads(path.read_text())
    return {r["policy"]: r for r in records}


def _panels(accuracy_metric: str) -> list[tuple[str, str, float, float]]:
    """(field, panel title, value scale, baseline) for the three metric panels.

    baseline is the value plotted where the metric is undefined for a policy: the
    uncached policies (None/Exact) have 0% hit rate and 1x speedup -- their floor,
    not a gap. Accuracy is always present, so its baseline is nan.
    """
    return [
        (accuracy_metric, ACC_TITLES[accuracy_metric], 1.0, np.nan),
        ("hit_rate", "Cache Hit Rate (%)", 100.0, 0.0),
        ("full_speedup_x", "Full-System Speedup (x)", 1.0, 1.0),
    ]


# Short filename slug per metric, used when writing one PDF per panel.
SLUGS = {"hit": "hits1", "accuracy": "accuracy", "f1": "f1",
         "hit_rate": "hit_rate", "full_speedup_x": "speedup"}


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",      # Matches IEEE Times/Computer Modern
        "font.size": 9,              # Standard font size for IEEE figures
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


def _policies(runs: dict[str, dict[str, dict]]) -> list[str]:
    return [p for p in POLICY_ORDER if any(p in runs[r] for r in runs)]


def _draw_panel(ax, runs, policies, field, title, scale, baseline) -> None:
    x = np.arange(len(policies))
    for ri, run_name in enumerate(runs):
        recs = runs[run_name]
        y = []
        for p in policies:
            v = recs.get(p, {}).get(field)
            # An undefined metric falls back to the panel baseline (0% hit rate,
            # 1x speedup) so the uncached policies show their floor, not a gap.
            y.append(v * scale if isinstance(v, (int, float)) else baseline)
        ax.plot(x, y, marker=MARKERS[ri % len(MARKERS)],
                linestyle=LINESTYLES[ri % len(LINESTYLES)],
                color=COLORS[ri % len(COLORS)], label=pretty_label(run_name))

    if field == "full_speedup_x":
        ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")  # no-speedup ref
    ax.set_title(title)
    ax.set_xlabel("Cache Policy")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies], rotation=30, ha="right")
    ax.grid(True, linestyle="--", alpha=0.6)


def _finish(fig, legend_ax, runs, ncol_cap: int, rect: list[float]) -> None:
    """Shared bottom legend for >1 run (as in characterization/), else a title."""
    run_names = list(runs.keys())
    if len(run_names) > 1:
        handles, lbls = legend_ax.get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=min(len(run_names), ncol_cap),
                   bbox_to_anchor=(0.5, -0.05), frameon=False)
        plt.tight_layout(rect=rect)
    else:
        fig.suptitle(pretty_label(run_names[0]), fontsize=9, y=1.02)
        plt.tight_layout()


def make_figure(runs: dict[str, dict[str, dict]], accuracy_metric: str) -> plt.Figure:
    """All three metrics in one double-column figure."""
    _set_style()
    policies = _policies(runs)
    panels = _panels(accuracy_metric)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 2.8))
    for ax, (field, title, scale, baseline) in zip(axes, panels):
        _draw_panel(ax, runs, policies, field, title, scale, baseline)
    _finish(fig, axes[0], runs, ncol_cap=5, rect=[0, 0.08, 1, 1])
    return fig


def make_separate(runs: dict[str, dict[str, dict]],
                  accuracy_metric: str) -> list[tuple[str, plt.Figure]]:
    """One single-column figure per metric; returns [(field, figure), ...]."""
    _set_style()
    policies = _policies(runs)
    out: list[tuple[str, plt.Figure]] = []
    for field, title, scale, baseline in _panels(accuracy_metric):
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        _draw_panel(ax, runs, policies, field, title, scale, baseline)
        # ncol=1: keep the legend column no wider than the single-column plot.
        _finish(fig, ax, runs, ncol_cap=1, rect=[0, 0.10, 1, 1])
        out.append((field, fig))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[],
                    help="tags, dirs, or summary.json paths to plot")
    ap.add_argument("--all", action="store_true",
                    help="plot every run under artifacts/rog_cache/*/summary.json")
    ap.add_argument("--accuracy-metric", default="hit",
                    choices=["hit", "accuracy", "f1"],
                    help="which metric fills the accuracy panel (default: hit = Hits@1)")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "artifacts" / "plots" / "rog_cache_results.pdf",
                    help="output PDF path (with --separate, the base name; each panel "
                         "is written to <stem>_<metric>.pdf)")
    ap.add_argument("--separate", action="store_true",
                    help="write one PDF per subplot instead of a single combined figure")
    args = ap.parse_args()

    run_paths: list[str] = list(args.runs)
    if args.all:
        run_paths += [p.parent.name for p in sorted(ROG_DIR.glob("*/summary.json"))]
    if not run_paths:
        raise SystemExit("nothing to plot: pass --runs <tag...> or --all")

    runs: dict[str, dict[str, dict]] = {}
    for r in dict.fromkeys(run_paths):  # dedupe, keep order
        path = resolve_summary(r)
        if not path.exists():
            print(f"[skip] no summary.json for {r!r} ({path})")
            continue
        runs[Path(path).parent.name] = load_run(path)
    if not runs:
        raise SystemExit("no valid summary.json files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.output.suffix.lstrip(".") or "pdf"
    print(f"plotted {len(runs)} run(s): {', '.join(runs)}")

    if args.separate:
        for field, fig in make_separate(runs, args.accuracy_metric):
            path = args.output.with_name(
                f"{args.output.stem}_{SLUGS.get(field, field)}{args.output.suffix}")
            fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
            print(f"wrote {path}")
    else:
        fig = make_figure(runs, args.accuracy_metric)
        fig.savefig(args.output, format=fmt, bbox_inches="tight", dpi=300)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
