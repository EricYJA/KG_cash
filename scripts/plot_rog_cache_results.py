"""Plot RoG (and ToG) cache-experiment results from summary.json files.

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

# A run varies along three independent factors (model x paradigm x backend), so
# each gets its own visual channel instead of one flat colour per run: readers can
# then compare "all Gemini" or "all ToG" without decoding the legend entry by entry.
#   hue   -> model    (tab10 blue/orange/green, a colourblind-safe set)
#   shade -> backend  (Oxigraph = light tint, Virtuoso = full-strength)
#   hatch -> paradigm (RoG = solid, ToG = diagonal)
MODEL_COLORS = {"Gemini": "#1f77b4", "Haiku": "#ff7f0e", "GPT": "#2ca02c"}
FALLBACK_COLOR = "#7f7f7f"
PARADIGM_HATCH = {"RoG": "", "ToG": "//"}
# Tint fraction for the light (Oxigraph) shade: blend this much white into the hue.
LIGHT_BLEND = 0.55
# Hatches are drawn in the edge colour, so bars get a dark outline rather than the
# house white one -- a white hatch is invisible on the light backend shade.
EDGE_COLOR = "#404040"

ACC_TITLES = {"hit": "Hits@1 (%)", "accuracy": "Accuracy (%)", "f1": "F1 (%)"}

FIG_WIDTH = 6.5
PANEL_HEIGHT = 2.5
SOLO_PANEL_HEIGHT = 3.0  # a metric written to its own figure gets more room

# The speedup panel plots two series on one axis rather than a single value: the
# per-hit ratio and the whole-workload one it implies. Drawn by _draw_speedup_pair.
#
#   per hit        speedup_x = avg_miss_s / avg_hit_s -- conditional on a hit, i.e.
#                  how much cheaper a cache-served question is than a cold one
#   whole workload 1 / ((1 - h) + h/speedup_x) -- the same measurement amortized
#                  over every question, recomputed per record by _amdahl_speedup
#
# Neither number can be read without the other, which is why they share a panel and
# why there is no way to plot one alone: the per-hit figure on its own looks like a
# system result, and the amortized one on its own looks like no result.
#
# The amortized value is recomputed rather than read from the stored full_speedup_x
# because only the recomputation is right on every row. The two are algebraically
# identical (they agree to <=0.005 wherever the stored row is self-consistent), but
# the recomputation is derived from two fields that always share one denominator.
# That matters for the ToG per-pass summaries this script plots by default (see
# _pass1_records): summarize_tog_cache.py divides those by the *whole run's* avg
# miss, not the pass's own, so their stored full_speedup_x can be off by ~0.1 --
# e.g. gemini_tog_cache_virtuoso_test_pass1/semantic_lru reads 1.01 stored against
# 1.11 recomputed. Amdahl's h is the right accelerated fraction here because the
# questions that hit cost 0.91-1.02x the run average in the uncached baseline.
PAIR_FIELD = "speedup_pair"

YLABELS = {"Speedup (x)": "Speedup",
           "Cache Hit Rate (%)": "Hit Rate (%)"}

DROP_NONE_FIELDS = {"hit_rate", PAIR_FIELD}
SET_YLIM_FIELDS = {"hit_rate"}

PAIR_DARKEN = 0.45
PAIR_WIDTH_FRAC = 0.52

PAIR_BAR_BASE = 0.0

PAIR_REFERENCE = 1.0

PAIR_MAX_LABELED_HITS = 4

AXES_WIDTH_FRAC = 0.85
PAIR_LABEL_SIZE = 7.0
PAIR_LABEL_MIN_SIZE = 5.0
PAIR_LABEL_CHARS = 4
PAIR_LABEL_ASPECT = 0.62
PAIR_LABEL_CAP = 1.35


def parse_tag(tag: str) -> tuple[str, str, str | None]:
    """(paradigm, model, backend) decoded from a run tag.

    paradigm: *tog* -> "ToG", else "RoG"; model: *gemini* -> "Gemini", *gpt* ->
    "GPT", else "Haiku"; backend: *virtuoso* -> "Virt", *oxi* -> "Oxi", else None.
    """
    t = tag.lower()
    paradigm = "ToG" if "tog" in t else "RoG"
    model = "Gemini" if "gemini" in t else "GPT" if "gpt" in t else "Haiku"
    backend = "Virt" if "virtuoso" in t else "Oxi" if "oxi" in t else None
    return paradigm, model, backend


def pretty_label(tag: str) -> str:
    """Compact legend label, e.g. "RoG Gemini (Oxi)"."""
    paradigm, model, backend = parse_tag(tag)
    label = f"{paradigm} {model}"
    if backend:
        label += f" ({backend})"
    return label


def _tint(hex_color: str, amount: float) -> str:
    """Blend `hex_color` `amount` of the way toward white (0 = unchanged, 1 = white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = tuple(int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return "#%02x%02x%02x" % mix


def _darken(hex_color: str, amount: float) -> str:
    """Blend `hex_color` `amount` of the way toward black (0 = unchanged, 1 = black)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = tuple(int(round(c * (1.0 - amount))) for c in (r, g, b))
    return "#%02x%02x%02x" % mix


def bar_style(tag: str) -> dict:
    """Bar kwargs for a run: hue by model, shade by backend, hatch by paradigm.

    Runs with an unrecognised backend get the full-strength hue, so a tag without a
    backend marker is never mistaken for the light (Oxigraph) member of a pair.
    """
    paradigm, model, backend = parse_tag(tag)
    color = MODEL_COLORS.get(model, FALLBACK_COLOR)
    if backend == "Oxi":
        color = _tint(color, LIGHT_BLEND)
    return {"color": color, "hatch": PARADIGM_HATCH.get(paradigm, "")}


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
        (PAIR_FIELD, "Speedup (x)", 1.0, 1.0),
    ]


# Short filename slug per metric, used when writing one PDF per panel.
SLUGS = {"hit": "hits1", "accuracy": "accuracy", "f1": "f1",
         "hit_rate": "hit_rate", "speedup_pair": "speedup"}


def _amdahl_speedup(record: dict, baseline: float) -> float:
    """Amdahl's law over one record: 1 / ((1 - h) + h / s).

    The cache accelerates only the fraction `h` of questions it serves, each by
    `s = speedup_x`; the remaining (1 - h) run at full cost. Returns `baseline`
    (1x) where the cache served nothing, and nan where the row never measured the
    inputs -- an unmeasured run must not appear as a solid "exactly 1.00x" bar.
    """
    h = record.get("hit_rate")
    s = record.get("speedup_x")
    if h is None or "speedup_x" not in record:
        return np.nan  # never measured
    if not h:
        return baseline  # no hits => nothing accelerated
    if not isinstance(s, (int, float)) or s <= 0:
        return np.nan  # hits but no usable per-hit speedup
    return 1.0 / ((1.0 - h) + h / s)


def _per_hit_speedup(record: dict, baseline: float) -> float:
    """avg_miss_s / avg_hit_s as the summarizer stored it, with the same floors."""
    s = record.get("speedup_x")
    if "speedup_x" not in record:
        return np.nan  # never measured
    if s is None:
        return baseline  # measured, undefined (no hits to time)
    if not isinstance(s, (int, float)) or s <= 0:
        return np.nan
    return float(s)


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",  # Matches the journal body font (Times/CM)
        "font.size": 10,  # Figure is printed at 100%, so match body text
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        # Only the axis lines the data needs: the ticks carry the scale, so the
        # top/right box is ink that frames nothing.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.6,  # thin enough not to muddy the fill colour
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


def _records_for(runs, run_name):
    """The per-policy records a panel should read for `run_name`.

    Every metric is reported for the 1st pass (cold cache) where a per-pass summary
    exists, so warm passes don't inflate hit rate / speedup / accuracy. Single-pass
    runs (all RoG) have no _pass1 summary and use the whole run.
    """
    pass1 = _pass1_records(run_name)
    return pass1 if pass1 is not None else runs[run_name]


def _draw_speedup_pair(ax, runs, policies, x, width, n, all_y: list[float]) -> None:
    """Per-hit and whole-workload speedup on one axis, two bars per run.

    These are the same measurement asked two different questions, and the paper's
    result is the *gap* between them: a cache hit answers a question ~2-3.7x faster,
    but the cache only serves ~7-8% of questions, so the workload as a whole moves
    ~1.04-1.06x. Plotted apart, each number invites the wrong reading -- the per-hit
    figure looks like a system result, the amortized one looks like no result.
    Plotted together against the 1x rule, the small bar in front of the large one
    *is* the finding: per-hit cost is not the bottleneck, coverage is.
    """
    rotation, fontsize = _pair_label_layout(n, len(policies))
    label_every_hit = len(runs) <= PAIR_MAX_LABELED_HITS
    per_run_hits: list[list[float]] = []  # [run][policy], for the per-group best
    for ri, run_name in enumerate(runs):
        src = _records_for(runs, run_name)
        offset = (ri - (n - 1) / 2.0) * width
        style = bar_style(run_name)
        per_hit, full = [], []
        for p in policies:
            record = src.get(p, {})
            per_hit.append(_per_hit_speedup(record, PAIR_REFERENCE))
            full.append(_amdahl_speedup(record, PAIR_REFERENCE))
        ax.bar(x + offset, per_hit, bottom=PAIR_BAR_BASE,
               width=width, label=pretty_label(run_name), edgecolor=EDGE_COLOR,
               linewidth=0.6, zorder=3, **style)
        ax.bar(x + offset, full, bottom=PAIR_BAR_BASE,
               width=width * PAIR_WIDTH_FRAC, color=_darken(style["color"], PAIR_DARKEN),
               edgecolor=EDGE_COLOR, linewidth=0.6, zorder=4)
        _label_pair(ax, x + offset, per_hit, full, rotation, fontsize,
                    label_per_hit=label_every_hit)
        per_run_hits.append(per_hit)
        all_y.extend(v for v in per_hit if isinstance(v, (int, float)) and np.isfinite(v))

    if not label_every_hit:
        # Too many runs to number every per-hit bar, but the tallest one in each
        # policy group is the figure's headline number, so it still gets its value.
        _label_best_per_group(ax, x, per_run_hits, width, n)


def _pair_label_layout(n_runs: int, n_policies: int) -> tuple[float, float]:
    """(rotation, fontsize) that keeps a whole-workload label inside its bar slot.

    Horizontal while the slot is wide enough to hold the digits; past that the label
    is turned on its side, where its footprint is the line height rather than the
    string length -- roughly a third as wide -- and shrunk further only if even that
    does not fit. Suppressing the label instead is not an option: it is the only
    place the whole-workload number appears.
    """
    slot_in = (FIG_WIDTH * AXES_WIDTH_FRAC) / max(n_policies, 1) * (0.8 / max(n_runs, 1))
    if slot_in >= PAIR_LABEL_CHARS * PAIR_LABEL_ASPECT * PAIR_LABEL_SIZE / 72.0:
        return 0.0, PAIR_LABEL_SIZE
    return 90.0, max(min(PAIR_LABEL_SIZE, slot_in * 72.0 / PAIR_LABEL_CAP),
                     PAIR_LABEL_MIN_SIZE)


def _label_pair(ax, xs, per_hit, full, rotation: float, fontsize: float,
                label_per_hit: bool) -> None:
    """Value labels: above the tall bar, and on top of the inset one.

    The whole-workload label needs an opaque backing because it lands inside the
    per-hit bar's fill, which may be any of the run hues; rotated, it reads bottom-up
    out of the sliver it belongs to.
    """
    for xc, hv, fv in zip(xs, per_hit, full):
        if label_per_hit and isinstance(hv, (int, float)) and np.isfinite(hv):
            ax.annotate(f"{hv:.2f}", (xc, hv), textcoords="offset points",
                        xytext=(0, 3), ha="center", va="bottom",
                        fontsize=PAIR_LABEL_SIZE, fontweight="bold", zorder=6)
        if isinstance(fv, (int, float)) and np.isfinite(fv):
            ax.annotate(f"{fv:.2f}", (xc, fv), textcoords="offset points",
                        xytext=(0, 2), ha="center", va="bottom", rotation=rotation,
                        rotation_mode="anchor", fontsize=fontsize,
                        fontweight="bold", zorder=6,
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                              "ec": "none", "alpha": 0.9})


def _draw_panel(ax, runs, policies, field, title, scale, baseline,
                reserve_legend_headroom: bool = True) -> None:
    """Grouped bar chart for one metric: one bar per run within each policy group.
"""
    x = np.arange(len(policies))
    n = max(len(runs), 1)
    width = 0.8 / n
    all_y: list[float] = []

    if field == PAIR_FIELD:
        _draw_speedup_pair(ax, runs, policies, x, width, n, all_y)
    else:
        for ri, run_name in enumerate(runs):
            src = _records_for(runs, run_name)
            y = []
            for p in policies:
                record = src.get(p, {})
                v = record.get(field)
                if isinstance(v, (int, float)):
                    y.append(v * scale)
                elif field in record:

                    y.append(baseline)
                else:

                    y.append(np.nan)
            all_y.extend(y)
            offset = (ri - (n - 1) / 2.0) * width
            ax.bar(x + offset, y, width=width, label=pretty_label(run_name),
                   edgecolor=EDGE_COLOR, linewidth=0.6, zorder=3,
                   **bar_style(run_name))

    finite = [v for v in all_y if isinstance(v, (int, float)) and np.isfinite(v)]
    if field == PAIR_FIELD:
        # Full 0-based bars, with the 1x rule carrying the "no change" reference the
        # floor would otherwise have marked. Top is the per-hit series' own range
        # (2-4x), left with headroom for the value labels.
        top = max(finite) if finite else PAIR_REFERENCE
        ax.set_ylim(PAIR_BAR_BASE, top / 0.82)
        ax.axhline(PAIR_REFERENCE, color="#555555", linewidth=0.9, linestyle=":",
                   zorder=5)
    elif field in SET_YLIM_FIELDS:
        # Hit rate keeps its real 0 baseline and auto-scales its top.
        hi = max(finite) if finite else 1.0
        if reserve_legend_headroom:
            # Headroom for the in-axes legend: more legend rows -> more whitespace.
            rows = (len(runs) + 1) // 2  # 2-column legend
            bar_frac = max(0.85 - 0.06 * rows, 0.45)  # fraction of axis for bars
        else:
            # No in-axes legend: just enough clearance for the value labels.
            bar_frac = 0.90
        ax.set_ylim(0.0, hi / bar_frac)
    # No panel title -- the metric names the y-axis instead, so a paper caption
    # carries the framing and the figure stays all data.
    ax.set_ylabel(YLABELS.get(title, title))
    ax.set_xlabel("Cache Policy")
    ax.set_xticks(x)
    # Labels are short enough to sit horizontal at this width -- rotating them only
    # costs the reader a head tilt.
    ax.set_xticklabels([POLICY_LABELS.get(p, p) for p in policies])
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.margins(x=0.02)


def _pair_proxies() -> list:
    """Two neutral swatches naming the series the combined speedup panel draws.
    """
    from matplotlib.patches import Patch
    return [Patch(facecolor="#b0b0b0", edgecolor=EDGE_COLOR, linewidth=0.6,
                  label="Per hit"),
            Patch(facecolor=_darken("#b0b0b0", PAIR_DARKEN), edgecolor=EDGE_COLOR,
                  linewidth=0.6, label="Whole workload")]


def _add_figure_legend(fig, src_ax, n_runs: int, extra: list | None = None) -> None:
    """Shared bold legend above the panels, built from `src_ax`'s handles.

    """
    n_entries = n_runs + len(extra or [])
    ncol = min(4, n_entries)
    rows = -(-n_entries // ncol)  # ceil
    reserved_in = 0.17 * rows + 0.20  # legend rows + frame padding
    top = 1.0 - reserved_in / fig.get_figheight()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))

    handles, labels = src_ax.get_legend_handles_labels()
    for patch in extra or []:
        handles.append(patch)
        labels.append(patch.get_label())
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, 1.0), ncol=ncol,
                     framealpha=0.9, edgecolor="#cccccc", fontsize=8,
                     handlelength=1.2, handletextpad=0.5, labelspacing=0.3,
                     columnspacing=1.0, borderpad=0.4)
    leg.get_frame().set_linewidth(0.8)
    for text in leg.get_texts():
        text.set_fontweight("bold")


def make_figure(runs: dict[str, dict[str, dict]], accuracy_metric: str,
                ) -> plt.Figure:
    """All three metrics in one journal-width figure, stacked one panel per row."""
    _set_style()
    policies = _policies(runs)
    panels = _panels(accuracy_metric)

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(FIG_WIDTH, PANEL_HEIGHT * len(panels)))
    for ax, (field, title, scale, baseline) in zip(axes, panels):
        panel_policies = [p for p in policies if p != "none"] \
            if field in DROP_NONE_FIELDS else policies
        _draw_panel(ax, runs, panel_policies, field, title, scale, baseline,
                    reserve_legend_headroom=False)
    plt.tight_layout()

    extra = _pair_proxies() if any(f == PAIR_FIELD for f, *_ in panels) else None
    if len(runs) > 1 or extra:
        _add_figure_legend(fig, axes[0], len(runs), extra)
    return fig


def make_separate(runs: dict[str, dict[str, dict]], accuracy_metric: str,
                  ) -> list[tuple[str, plt.Figure]]:
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


        extra = _pair_proxies() if field == PAIR_FIELD else None
        if len(runs) > 1 or extra:
            _add_figure_legend(fig, ax, len(runs), extra)
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
