#!/usr/bin/env python3
"""Plot ToG cache-experiment results, one figure per loop (pass).

The ToG twin of plot_rog_cache_results.py. That script overlays a run's cold
(pass 1) and warm (pass 2) lines on the *same* panels; this one instead draws a
**separate figure per loop** -- loop 1 (cold cache) in one plot, loop 2 (warm
cache) in another -- so the per-loop behaviour is read without the two passes
crowding each other. Within each loop figure the layout matches the RoG figures
exactly: three metric panels (accuracy / cache hit rate / full-system speedup),
policies on the x-axis, one line per run, shared bottom legend, IEEE serif style.

Input is the per-pass summaries written by

    scripts/summarize_tog_cache.py --run <tag> --per-pass

which land in artifacts/tog_cache/<tag>_pass<K>/summary.json (same schema as the
RoG summaries). Pass a *base* tag with --runs and both of its _pass<K> summaries
are picked up automatically and split across the loop figures.

    # one ToG run -> loop-1 and loop-2 figures
    python scripts/plot_tog_cache_results.py --runs gemini_tog_cache_oxi_test

    # several runs overlaid, still split by loop
    python scripts/plot_tog_cache_results.py \
        --runs gemini_tog_cache_oxi_test gemini_tog_cache_virtuoso_test

    # everything with per-pass summaries under artifacts/tog_cache/
    python scripts/plot_tog_cache_results.py --all

Each loop K is written to <output-stem>_loop<K><suffix> (default base:
artifacts/plots/tog_cache_results.pdf -> ..._loop1.pdf, ..._loop2.pdf).
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TOG_DIR = REPO_ROOT / "artifacts" / "tog_cache"

# A per-pass summary dir is named "<base tag>_pass<K>" (K is 1-based; see
# scripts/summarize_tog_cache.py). We group summaries by K into one figure each.
PASS_RE = re.compile(r"^(?P<base>.+)_pass(?P<k>\d+)$")


def _load_rog_module():
    """Load plot_rog_cache_results.py by path and reuse its plotting internals.

    scripts/ is not a package, so import by file. The module only defines helpers
    behind a __main__ guard, so importing it runs no plotting.
    """
    path = Path(__file__).resolve().parent / "plot_rog_cache_results.py"
    spec = importlib.util.spec_from_file_location("plot_rog_cache_results", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rog = _load_rog_module()

# Panels where the uncached 'none' policy is a trivial floor (0% hit rate, ~1x
# speedup), so --drop-none leaves it off. Accuracy keeps 'none' as a real baseline.
DROP_NONE_FIELDS = {"hit_rate", "full_speedup_x"}


def _draw_bar_panel(ax, runs, policies, field, title, scale, baseline) -> None:
    """Grouped bar chart for one metric: one bar per run within each policy group.

    The bar twin of rog._draw_panel (which draws lines). Same colour-per-run and
    baseline-fallback semantics, styled to match the repo's other bar charts
    (characterization/): serif IEEE rcParams from rog._set_style, white bar edges,
    a dashed y-grid behind the bars. An in-axes legend keeps the plot dominant.

    For the Full-System Speedup panel the y-axis is zoomed to start just below the
    1x reference (rather than 0) so the near-1x spread on the cold pass -- and the
    per-policy differences on the warm pass -- are actually legible.
    """
    x = np.arange(len(policies))
    n = max(len(runs), 1)
    width = 0.8 / n                       # groups span 0.8 of the unit slot
    all_y: list[float] = []
    for ri, run_name in enumerate(runs):
        recs = runs[run_name]
        y = []
        for p in policies:
            v = recs.get(p, {}).get(field)
            # Undefined metric -> panel baseline (0% hit rate, 1x speedup), matching
            # the line version so uncached policies show their floor, not a gap.
            y.append(v * scale if isinstance(v, (int, float)) else baseline)
        all_y.extend(y)
        offset = (ri - (n - 1) / 2.0) * width
        ax.bar(x + offset, y, width=width, color=rog.COLORS[ri % len(rog.COLORS)],
               label=rog.pretty_label(run_name), edgecolor="white", linewidth=0.8,
               zorder=3)

    if field == "full_speedup_x":
        ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
        # y-axis runs 0.5 -> max bar: the near-1x differences stay legible and the
        # empty band below the 1x reference gives the legend a place to sit.
        hi = max(all_y) if all_y else 1.0
        ax.set_ylim(0.5, hi + max((hi - 1.0) * 0.10, 0.05))
    ax.set_title(title)
    ax.set_xlabel("Cache Policy")
    ax.set_xticks(x)
    ax.set_xticklabels([rog.POLICY_LABELS.get(p, p) for p in policies],
                       rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.margins(x=0.02)


def loop_runs(run_args: list[str], all_flag: bool) -> dict[int, dict[str, dict]]:
    """Group per-pass summaries by loop index K -> {base tag: {policy: record}}.

    A --runs value is a *base* tag whose _pass<K> summaries are globbed in (so one
    tag feeds every loop figure); a dir or direct summary.json path also works and
    is read as-is. --all sweeps every artifacts/tog_cache/*_pass<K>/summary.json.
    """
    pass_tags: list[str] = []
    if all_flag:
        pass_tags += [p.parent.name for p in sorted(TOG_DIR.glob("*_pass*/summary.json"))]
    for run in run_args:
        globbed = sorted(TOG_DIR.glob(f"{run}_pass*/summary.json"))
        # A bare base tag expands to its passes; otherwise treat it as an explicit
        # _pass<K> tag / dir / summary.json path and resolve it directly.
        pass_tags += [p.parent.name for p in globbed] if globbed else [run]

    loops: dict[int, dict[str, dict]] = {}
    seen: set[Path] = set()
    for tag in pass_tags:
        path = rog.resolve_summary(tag, TOG_DIR)
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            print(f"[skip] no summary.json for {tag!r} ({path})")
            continue
        name = Path(path).parent.name
        m = PASS_RE.match(name)
        if not m:
            print(f"[skip] {name!r}: not a _pass<K> summary (run summarize_tog_cache "
                  f"--per-pass first)")
            continue
        base, k = m.group("base"), int(m.group("k"))
        # Key by base tag so rog.pretty_label yields a clean legend label without a
        # redundant "pass N" -- the loop is already the figure's subject.
        loops.setdefault(k, {})[base] = rog.load_run(path)
    return loops


def make_loop_figure(runs: dict[str, dict], accuracy_metric: str, loop_idx: int,
                     drop_none: bool = True) -> plt.Figure:
    """One RoG-style three-panel figure for a single loop, all runs overlaid.

    When ``drop_none`` is set (the default), the uncached ``none`` policy is left
    off the Cache Hit Rate and Full-System Speedup panels: it is a trivial floor
    there (0% hits, ~1x speedup), so keeping it only stretches the x-axis. The
    accuracy panel still shows every policy.
    """
    rog._set_style()
    policies = rog._policies(runs)
    panels = rog._panels(accuracy_metric)
    fig, axes = plt.subplots(1, len(panels), figsize=(14.0, 5.0))
    speedup_ax = axes[0]
    for ax, (field, title, scale, baseline) in zip(axes, panels):
        panel_policies = policies
        if field in DROP_NONE_FIELDS and drop_none:
            panel_policies = [p for p in policies if p != "none"]
        _draw_bar_panel(ax, runs, panel_policies, field, title, scale, baseline)
        if field == "full_speedup_x":
            speedup_ax = ax

    # No suptitle -- keep the panels the whole figure. The legend sits in the empty
    # band below the 1x line on the speedup panel, so it steals no reserved margin.
    if len(runs) > 1:
        speedup_ax.legend(loc="lower center", framealpha=0.9, fontsize=9, ncol=2,
                          handlelength=1.2, handletextpad=0.5, labelspacing=0.3,
                          columnspacing=1.0, borderpad=0.4)
    plt.tight_layout()
    return fig


def make_loop_separate(runs: dict[str, dict], accuracy_metric: str, loop_idx: int,
                       drop_none: bool = True) -> list[tuple[str, plt.Figure]]:
    """One single-column figure per metric for a single loop; [(field, fig), ...].

    The per-loop twin of rog.make_separate: same three metrics, but each written to
    its own figure so the panels can be placed independently. ``drop_none`` keeps
    the uncached ``none`` policy off the Cache Hit Rate and Full-System Speedup
    figures (a trivial floor on both), as in the combined layout.
    """
    rog._set_style()
    policies = rog._policies(runs)
    out: list[tuple[str, plt.Figure]] = []
    for field, title, scale, baseline in rog._panels(accuracy_metric):
        panel_policies = policies
        if field in DROP_NONE_FIELDS and drop_none:
            panel_policies = [p for p in policies if p != "none"]
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        _draw_bar_panel(ax, runs, panel_policies, field, title, scale, baseline)
        # No suptitle; in-axes legend so the bars stay the dominant element. On the
        # speedup figure it drops into the empty band below the 1x reference.
        if len(runs) > 1:
            loc = "lower center" if field == "full_speedup_x" else "best"
            ax.legend(loc=loc, framealpha=0.9, fontsize=9, ncol=2, handlelength=1.2,
                      handletextpad=0.5, labelspacing=0.3, columnspacing=1.0,
                      borderpad=0.4)
        plt.tight_layout()
        out.append((field, fig))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[],
                    help="ToG base tags (under artifacts/tog_cache/); each expands "
                         "to its _pass<K> summaries. Dirs / summary.json paths also work.")
    ap.add_argument("--all", action="store_true",
                    help="plot every artifacts/tog_cache/*_pass<K>/summary.json")
    ap.add_argument("--accuracy-metric", default="hit",
                    choices=["hit", "accuracy", "f1"],
                    help="which metric fills the accuracy panel (default: hit = Hits@1)")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "artifacts" / "plots" / "tog_cache_results.pdf",
                    help="output base path; each loop is written to "
                         "<stem>_loop<K><suffix>")
    ap.add_argument("--drop-none", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="omit the uncached 'none' policy from the Cache Hit Rate "
                         "and Full-System Speedup panels (a trivial 0%%/1x floor on "
                         "both); use --no-drop-none to keep it")
    ap.add_argument("--separate", action="store_true",
                    help="write one figure per metric per loop "
                         "(<stem>_loop<K>_<metric><suffix>) instead of a single "
                         "combined 3-panel figure per loop")
    args = ap.parse_args()

    if not args.runs and not args.all:
        raise SystemExit("nothing to plot: pass --runs and/or --all")

    loops = loop_runs(args.runs, args.all)
    if not loops:
        raise SystemExit(
            "no per-pass ToG summaries found. Generate them with:\n"
            "  scripts/summarize_tog_cache.py --run <tag> --per-pass")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fmt = args.output.suffix.lstrip(".") or "pdf"
    for loop_idx in sorted(loops):
        runs = loops[loop_idx]
        if args.separate:
            for field, fig in make_loop_separate(runs, args.accuracy_metric, loop_idx,
                                                  drop_none=args.drop_none):
                path = args.output.with_name(
                    f"{args.output.stem}_loop{loop_idx}_"
                    f"{rog.SLUGS.get(field, field)}{args.output.suffix}")
                fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
                plt.close(fig)
                print(f"wrote {path}  (loop {loop_idx}: {', '.join(runs)})")
        else:
            fig = make_loop_figure(runs, args.accuracy_metric, loop_idx,
                                   drop_none=args.drop_none)
            path = args.output.with_name(f"{args.output.stem}_loop{loop_idx}{args.output.suffix}")
            fig.savefig(path, format=fmt, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"wrote {path}  (loop {loop_idx}: {', '.join(runs)})")


if __name__ == "__main__":
    main()
