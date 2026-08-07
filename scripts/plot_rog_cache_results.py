#!/usr/bin/env python3
"""Plot RoG (and ToG) cache-experiment results from summary.json files.

Each run's summary.json (RoG: summarize_rog_cache.py; ToG: summarize_tog_cache.py)
is a list of per-policy records with: policy, hit (Hits@1), f1, accuracy,
hit_rate, speedup_x, full_speedup_x. Policies go on the x-axis and each metric
gets its own panel (small multiples -- never a dual y-axis). One grouped bar per
run within each policy, styled to match the characterization/ figures (IEEE serif,
tab10 colours, white bar edges, dashed y-grid, shared bottom legend, PDF). ToG runs
(--tog-runs) overlay on the same panels as additional bars in each group.

    # one RoG run
    python scripts/plot_rog_cache_results.py --runs rog_cache_virtuoso_test

    # RoG vs ToG on the same figure
    python scripts/plot_rog_cache_results.py \
        --runs gemini_rog_cache_oxi_test --tog-runs gemini_tog_cache_oxi_test

    # everything found under artifacts/{rog,tog}_cache/
    python scripts/plot_rog_cache_results.py --all --tog-all

A --runs value resolves to artifacts/rog_cache/<tag>/summary.json (--tog-runs to
artifacts/tog_cache/<tag>/summary.json); a directory or direct summary.json path
also works for either.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ROG_DIR = REPO_ROOT / "artifacts" / "rog_cache"
TOG_DIR = REPO_ROOT / "artifacts" / "tog_cache"

# Fixed policy order + short labels (baseline first, then caching policies).
POLICY_ORDER = ["none", "exact", "semantic_lfu", "semantic_lru", "semantic_oracle"]
POLICY_LABELS = {
    "none": "None",
    "exact": "Exact",
    "semantic_lfu": "Sem-LFU",
    "semantic_lru": "Sem-LRU",
    "semantic_oracle": "Sem-Oracle",
}

# characterization/ house style: full tab10 (10 entries) so overlaying RoG+ToG
# across both backends -- up to 10 runs, e.g. --all -- never wraps two runs onto
# the same colour.
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

ACC_TITLES = {"hit": "Hits@1 (%)", "accuracy": "Accuracy (%)", "f1": "F1 (%)"}

# Panels that get the same treatment: 'none' dropped (a trivial floor there --
# 0% hits / 1x speedup), headroom reserved above the bars, and a bold legend
# tucked into that whitespace. The accuracy panel keeps 'none' and its "best"
# legend placement, since there 'none' is a real baseline.
DROP_NONE_FIELDS = {"hit_rate", "full_speedup_x"}
LEGEND_ABOVE_FIELDS = {"hit_rate", "full_speedup_x"}


def pretty_label(tag: str) -> str:
    """Compact legend label distinguishing paradigm, model, and backend.

    paradigm: *tog* -> "ToG", else "RoG"; model: gemini* -> "Gemini", else "Haiku";
    backend: *virtuoso* -> "Virt", *oxi* -> "Oxi". e.g. "RoG Gemini (Oxi)".
    """
    t = tag.lower()
    paradigm = "ToG" if "tog" in t else "RoG"
    model = "Gemini" if "gemini" in t else "Haiku"
    backend = "Virt" if "virtuoso" in t else "Oxi" if "oxi" in t else None
    label = f"{paradigm} {model}"
    if backend:
        label += f" ({backend})"
    return label


# Per-pass ToG summaries (…_pass1/…_pass2) are the loop-split plots' input, not
# whole-run lines; --all/--tog-all skip them (use --tog-runs to plot passes).
PASS_TAG_RE = re.compile(r"_pass\d+$")


def glob_runs(base: Path, *, exclude_gpt: bool = True, exclude_pass: bool = True) -> list[str]:
    """Run tags with a summary.json under `base`, minus GPT and per-pass variants.

    GPT runs are excluded so --all stays a Gemini/Haiku (both Virtuoso and Oxigraph)
    comparison; pass exclude_gpt=False to include them.
    """
    out = []
    for p in sorted(base.glob("*/summary.json")):
        name = p.parent.name
        if exclude_gpt and "gpt" in name.lower():
            continue
        if exclude_pass and PASS_TAG_RE.search(name):
            continue
        out.append(name)
    return out


def resolve_summary(run: str, base: Path = ROG_DIR) -> Path:
    """Accept a tag (resolved under `base`), a dir, or a direct summary.json path."""
    p = Path(run)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "summary.json"
    return base / run / "summary.json"


def load_run(path: Path) -> dict[str, dict]:
    """Return {policy: record} for one run's summary.json."""
    records = json.loads(path.read_text())
    return {r["policy"]: r for r in records}


# Every panel reports 1st-pass (cold-cache) numbers: a run's whole-run summary
# aggregates every loop pass, and the warm passes (~100% hit) inflate hit rate,
# speedup, and accuracy. For runs with a per-pass ToG summary we read pass 1
# instead; single-pass runs (all RoG) already are pass 1 and fall through unchanged.
_PASS1_CACHE: dict[str, dict[str, dict] | None] = {}


def _pass1_records(run_name: str) -> dict[str, dict] | None:
    """{policy: record} from run_name's _pass1 summary, or None if it has none."""
    if run_name not in _PASS1_CACHE:
        path = TOG_DIR / f"{run_name}_pass1" / "summary.json"
        _PASS1_CACHE[run_name] = load_run(path) if path.exists() else None
    return _PASS1_CACHE[run_name]


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
        "font.size": 17,             # Enlarged for readability
        "axes.labelsize": 19,
        "axes.titlesize": 19,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def _policies(runs: dict[str, dict[str, dict]]) -> list[str]:
    return [p for p in POLICY_ORDER if any(p in runs[r] for r in runs)]


def _draw_panel(ax, runs, policies, field, title, scale, baseline) -> None:
    """Grouped bar chart for one metric: one bar per run within each policy group.

    Styled to match the characterization/ bar figures: colour-per-run, white bar
    edges, a dashed y-grid behind the bars. The Full-System Speedup panel zooms its
    y-axis to start just below the 1x reference (not 0) so per-policy differences --
    which cluster near 1x -- stay legible instead of collapsing onto the floor.
    """
    x = np.arange(len(policies))
    n = max(len(runs), 1)
    width = 0.8 / n                       # groups span 0.8 of the unit slot
    all_y: list[float] = []
    for ri, run_name in enumerate(runs):
        recs = runs[run_name]
        # Every metric is reported for the 1st pass (cold cache) where a per-pass
        # summary exists, so warm passes don't inflate hit rate / speedup / accuracy.
        # Single-pass runs (all RoG) have no _pass1 summary and use the whole run.
        pass1 = _pass1_records(run_name)
        src = pass1 if pass1 is not None else recs
        y = []
        for p in policies:
            v = src.get(p, {}).get(field)
            # An undefined metric falls back to the panel baseline (0% hit rate,
            # 1x speedup) so the uncached policies show their floor, not a gap.
            y.append(v * scale if isinstance(v, (int, float)) else baseline)
        all_y.extend(y)
        offset = (ri - (n - 1) / 2.0) * width
        ax.bar(x + offset, y, width=width, color=COLORS[ri % len(COLORS)],
               label=pretty_label(run_name), edgecolor="white", linewidth=0.8,
               zorder=3)

    if field in LEGEND_ABOVE_FIELDS:
        # Reserve headroom above the tallest bar for the legend that sits on top --
        # more legend rows -> more whitespace. Speedup also zooms around its 1x
        # reference (bottom just below it); hit rate keeps its real 0 baseline.
        hi = max(all_y) if all_y else 1.0
        rows = (len(runs) + 1) // 2                     # 2-column legend
        bar_frac = max(0.85 - 0.06 * rows, 0.45)        # fraction of axis for bars
        if field == "full_speedup_x":
            ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)  # no-speedup ref
            lo = 0.9
        else:
            lo = 0.0
        ax.set_ylim(lo, lo + (hi - lo) / bar_frac)
    ax.set_title(title)
    ax.set_xlabel("Cache Policy")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies], rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.margins(x=0.02)


def _add_legend(ax, loc: str = "best") -> None:
    """Compact, bold, 2-column in-axes legend (no suptitle, no reserved band).

    Drawn inside ``ax`` so it steals no margin; the panels stay the dominant
    element. The hit-rate and speedup panels pass loc="upper center" to sit it in
    the whitespace reserved above their bars in _draw_panel.
    """
    leg = ax.legend(loc=loc, framealpha=0.9, edgecolor="#cccccc",
                    fontsize=9, ncol=2, handlelength=1.2, handletextpad=0.5,
                    labelspacing=0.3, columnspacing=1.0, borderpad=0.4)
    leg.get_frame().set_linewidth(0.8)
    for text in leg.get_texts():
        text.set_fontweight("bold")


def make_figure(runs: dict[str, dict[str, dict]], accuracy_metric: str) -> plt.Figure:
    """All three metrics in one double-column figure."""
    _set_style()
    policies = _policies(runs)
    panels = _panels(accuracy_metric)
    fig, axes = plt.subplots(1, len(panels), figsize=(14.0, 5.0))
    for ax, (field, title, scale, baseline) in zip(axes, panels):
        panel_policies = [p for p in policies if p != "none"] \
            if field in DROP_NONE_FIELDS else policies
        _draw_panel(ax, runs, panel_policies, field, title, scale, baseline)
        # Bold legend in the whitespace above the bars on the hit-rate and
        # speedup panels; the accuracy panel carries none.
        if field in LEGEND_ABOVE_FIELDS and len(runs) > 1:
            _add_legend(ax, loc="upper center")
    plt.tight_layout()
    return fig


def make_separate(runs: dict[str, dict[str, dict]],
                  accuracy_metric: str) -> list[tuple[str, plt.Figure]]:
    """One single-column figure per metric; returns [(field, figure), ...]."""
    _set_style()
    policies = _policies(runs)
    out: list[tuple[str, plt.Figure]] = []
    for field, title, scale, baseline in _panels(accuracy_metric):
        panel_policies = [p for p in policies if p != "none"] \
            if field in DROP_NONE_FIELDS else policies
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        _draw_panel(ax, runs, panel_policies, field, title, scale, baseline)
        if len(runs) > 1:
            loc = "upper center" if field in LEGEND_ABOVE_FIELDS else "best"
            _add_legend(ax, loc=loc)
        plt.tight_layout()
        out.append((field, fig))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[],
                    help="RoG runs: tags (under artifacts/rog_cache/), dirs, or "
                         "summary.json paths")
    ap.add_argument("--all", action="store_true",
                    help="plot every non-GPT run under artifacts/{rog,tog}_cache/ "
                         "(both Virtuoso and Oxigraph); per-pass ToG summaries are "
                         "skipped -- use --tog-runs for those")
    ap.add_argument("--tog-runs", nargs="+", default=[],
                    help="ToG runs: tags (under artifacts/tog_cache/), dirs, or "
                         "summary.json paths (from scripts/summarize_tog_cache.py). "
                         "Overlaid on the same panels as the RoG runs.")
    ap.add_argument("--tog-all", action="store_true",
                    help="plot every non-GPT, non-per-pass run under "
                         "artifacts/tog_cache/*/summary.json")
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

    # (run string, base dir) pairs so RoG tags resolve under artifacts/rog_cache
    # and ToG tags under artifacts/tog_cache; explicit paths ignore the base.
    pending: list[tuple[str, Path]] = [(r, ROG_DIR) for r in args.runs]
    # --all is the umbrella: every non-GPT run from both backends (RoG + ToG),
    # across Virtuoso and Oxigraph, minus the per-pass ToG loop summaries.
    if args.all:
        pending += [(n, ROG_DIR) for n in glob_runs(ROG_DIR)]
        pending += [(n, TOG_DIR) for n in glob_runs(TOG_DIR)]
    pending += [(r, TOG_DIR) for r in args.tog_runs]
    if args.tog_all:
        pending += [(n, TOG_DIR) for n in glob_runs(TOG_DIR)]
    if not pending:
        raise SystemExit("nothing to plot: pass --runs/--all and/or --tog-runs/--tog-all")

    runs: dict[str, dict[str, dict]] = {}
    seen: set = set()
    for r, base in pending:
        path = resolve_summary(r, base)
        if path in seen:
            continue
        seen.add(path)
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
