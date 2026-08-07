#!/usr/bin/env python3
"""Plot RoG (and ToG) cache-experiment results from summary.json files.

Each run's summary.json (RoG: summarize_rog_cache.py; ToG: summarize_tog_cache.py)
is a list of per-policy records with: policy, hit (Hits@1), f1, accuracy,
hit_rate, speedup_x, full_speedup_x. Policies go on the x-axis and each metric
gets its own panel (small multiples -- never a dual y-axis). One grouped bar per
run within each policy, styled to match the characterization/ figures (serif,
tab10 colours, white bar edges, dashed y-grid, shared bottom legend, PDF). ToG runs
(--tog-runs) overlay on the same panels as additional bars in each group.

Figures are sized for a single-column *journal* body (FIG_WIDTH-inch text block),
so panels stack vertically -- one metric per row, never side by side -- and the
fonts are set for a figure printed at its natural size rather than shrunk into an
IEEE two-column layout.

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
# The exact-match policy is deliberately absent: it is a near-zero-hit-rate
# degenerate case, so only the semantic policies are plotted.
POLICY_ORDER = ["none", "semantic_lfu", "semantic_lru", "semantic_oracle"]
POLICY_LABELS = {
    "none": "None",
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

# Journal (single-column body) sizing. FIG_WIDTH is a standard LaTeX article
# \textwidth, so a figure is placed at 100% and never scaled down -- which is why
# the fonts below are body-text sized instead of the inflated IEEE ones.
# Multi-metric figures stack PANEL_HEIGHT-inch rows; drop PANEL_HEIGHT to ~3.0 if
# the panels need to be shorter still.
FIG_WIDTH = 6.5
PANEL_HEIGHT = 2.5
SOLO_PANEL_HEIGHT = 3.0      # a metric written to its own figure gets more room

# Panels carry no title; the metric goes on the y-axis, where the long panel
# names don't fit rotated, so they get a shorter form.
YLABELS = {"Full-System Speedup (x)": "Speedup (x)",
           "Cache Hit Rate (%)": "Hit Rate (%)"}

# Panels that get the same treatment: 'none' dropped (a trivial floor there --
# 0% hits / 1x speedup) and an explicitly set y-range. The accuracy panel keeps
# 'none', since there it is a real baseline, and its automatic y-range.
DROP_NONE_FIELDS = {"hit_rate", "full_speedup_x"}
SET_YLIM_FIELDS = {"hit_rate", "full_speedup_x"}


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
    uncached policy (None) has 0% hit rate and 1x speedup -- its floor, not a gap.
    Accuracy is always present, so its baseline is nan.
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
        "font.family": "serif",      # Matches the journal body font (Times/CM)
        "font.size": 10,             # Figure is printed at 100%, so match body text
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
    })


def _policies(runs: dict[str, dict[str, dict]]) -> list[str]:
    return [p for p in POLICY_ORDER if any(p in runs[r] for r in runs)]


def _label_best_per_group(ax, x, per_run_y, width, n) -> None:
    """Annotate the tallest bar in each policy group with its value.

    Used on the speedup panels: the winning run per policy is the number readers
    quote, and with several runs per group only that one is worth the ink.
    """
    for j, xc in enumerate(x):
        col = [(ys[j], ri) for ri, ys in enumerate(per_run_y)
               if isinstance(ys[j], (int, float)) and np.isfinite(ys[j])]
        if not col:
            continue
        best, ri = max(col)
        offset = (ri - (n - 1) / 2.0) * width
        ax.annotate(f"{best:.2f}", (xc + offset, best), textcoords="offset points",
                    xytext=(0, 3), ha="center", va="bottom", fontsize=8,
                    fontweight="bold", zorder=4)


def _draw_panel(ax, runs, policies, field, title, scale, baseline,
                reserve_legend_headroom: bool = True) -> None:
    """Grouped bar chart for one metric: one bar per run within each policy group.

    Styled to match the characterization/ bar figures: colour-per-run, white bar
    edges, a dashed y-grid behind the bars. The Full-System Speedup panel zooms its
    y-axis to start just below the 1x reference (not 0) so per-policy differences --
    which cluster near 1x -- stay legible instead of collapsing onto the floor.

    ``reserve_legend_headroom`` pads the y-axis to leave a band above the bars for
    an in-axes legend. The stacked figure puts one shared legend above all panels
    instead, and passes False -- at these panel heights an in-axes legend would
    cover the bars rather than sit beside them.
    """
    x = np.arange(len(policies))
    n = max(len(runs), 1)
    width = 0.8 / n                       # groups span 0.8 of the unit slot
    all_y: list[float] = []
    per_run_y: list[list[float]] = []     # [run][policy], for the per-policy best label
    for ri, run_name in enumerate(runs):
        recs = runs[run_name]
        # Every metric is reported for the 1st pass (cold cache) where a per-pass
        # summary exists, so warm passes don't inflate hit rate / speedup / accuracy.
        # Single-pass runs (all RoG) have no _pass1 summary and use the whole run.
        pass1 = _pass1_records(run_name)
        src = pass1 if pass1 is not None else recs
        y = []
        for p in policies:
            record = src.get(p, {})
            v = record.get(field)
            if isinstance(v, (int, float)):
                y.append(v * scale)
            elif field in record:
                # Present but null: the metric was computed and is undefined for
                # this policy (no hits => no speedup to report). That is the
                # panel baseline -- 0% hit rate, 1x speedup -- a real floor.
                y.append(baseline)
            else:
                # Key absent: never measured. Runs predating the stage-2 timing
                # sidecar have no end-to-end speedup at all, and drawing them at
                # the baseline would put an unmeasured run on the chart as a
                # solid "exactly 1.00x" bar no reader could tell from a result.
                y.append(np.nan)
        all_y.extend(y)
        per_run_y.append(y)
        offset = (ri - (n - 1) / 2.0) * width
        ax.bar(x + offset, y, width=width, color=COLORS[ri % len(COLORS)],
               label=pretty_label(run_name), edgecolor="white", linewidth=0.8,
               zorder=3)

    if field == "full_speedup_x":
        _label_best_per_group(ax, x, per_run_y, width, n)

    if field in SET_YLIM_FIELDS:
        # Speedup zooms around its 1x reference (bottom just below it); hit rate
        # keeps its real 0 baseline.
        finite = [v for v in all_y if isinstance(v, (int, float)) and np.isfinite(v)]
        hi = max(finite) if finite else 1.0
        if field == "full_speedup_x":
            ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)  # no-speedup ref
            lo = 0.9
        else:
            lo = 0.0
        if reserve_legend_headroom:
            # Headroom for the in-axes legend: more legend rows -> more whitespace.
            rows = (len(runs) + 1) // 2                 # 2-column legend
            bar_frac = max(0.85 - 0.06 * rows, 0.45)    # fraction of axis for bars
        else:
            # No in-axes legend: just enough clearance for the value labels.
            bar_frac = 0.90
        ax.set_ylim(lo, lo + (hi - lo) / bar_frac)
    # No panel title -- the metric names the y-axis instead, so a paper caption
    # carries the framing and the figure stays all data.
    ax.set_ylabel(YLABELS.get(title, title))
    ax.set_xlabel("Cache Policy")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies], rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.margins(x=0.02)


def _add_figure_legend(fig, src_ax, n_runs: int) -> None:
    """Shared bold legend above the panels, built from `src_ax`'s handles.

    Sits outside the axes so it steals no plotting area from the (short) panels.
    Up to 4 columns, so ten runs land in three rows rather than a tall block.
    Used by both layouts: an in-axes legend needs headroom proportional to its own
    physical height, which at these panel heights means covering the bars.

    The band it occupies has to be reserved by re-running tight_layout with a
    ``rect`` first -- a figure legend is not laid out by tight_layout, so without
    this it is drawn straight over the top panel.
    """
    ncol = min(4, n_runs)
    rows = -(-n_runs // ncol)                    # ceil
    reserved_in = 0.17 * rows + 0.20             # legend rows + frame padding
    top = 1.0 - reserved_in / fig.get_figheight()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))

    handles, labels = src_ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), ncol=ncol,
                     framealpha=0.9, edgecolor="#cccccc", fontsize=8,
                     handlelength=1.2, handletextpad=0.5, labelspacing=0.3,
                     columnspacing=1.0, borderpad=0.4)
    leg.get_frame().set_linewidth(0.8)
    for text in leg.get_texts():
        text.set_fontweight("bold")


def make_figure(runs: dict[str, dict[str, dict]], accuracy_metric: str) -> plt.Figure:
    """All three metrics in one journal-width figure, stacked one panel per row."""
    _set_style()
    policies = _policies(runs)
    panels = _panels(accuracy_metric)
    # One metric per row: at a single-column journal width there is no room for
    # three panels side by side, and stacking keeps each x-axis readable.
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(FIG_WIDTH, PANEL_HEIGHT * len(panels)))
    for ax, (field, title, scale, baseline) in zip(axes, panels):
        panel_policies = [p for p in policies if p != "none"] \
            if field in DROP_NONE_FIELDS else policies
        _draw_panel(ax, runs, panel_policies, field, title, scale, baseline,
                    reserve_legend_headroom=False)
    plt.tight_layout()
    # One shared legend above the stack rather than a copy inside two panels: every
    # panel shows the same runs, and at this panel height an in-axes legend covers
    # the bars. bbox_inches="tight" at save time keeps it from being clipped.
    if len(runs) > 1:
        _add_figure_legend(fig, axes[0], len(runs))
    return fig


def make_separate(runs: dict[str, dict[str, dict]],
                  accuracy_metric: str) -> list[tuple[str, plt.Figure]]:
    """One journal-width figure per metric; returns [(field, figure), ...]."""
    _set_style()
    policies = _policies(runs)
    out: list[tuple[str, plt.Figure]] = []
    for field, title, scale, baseline in _panels(accuracy_metric):
        panel_policies = [p for p in policies if p != "none"] \
            if field in DROP_NONE_FIELDS else policies
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, SOLO_PANEL_HEIGHT))
        _draw_panel(ax, runs, panel_policies, field, title, scale, baseline,
                    reserve_legend_headroom=False)
        plt.tight_layout()
        # Legend above the axes, as in the stacked figure -- a single panel this
        # short has no in-axes whitespace big enough to hold it.
        if len(runs) > 1:
            _add_figure_legend(fig, ax, len(runs))
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
