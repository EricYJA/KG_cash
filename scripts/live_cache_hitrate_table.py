#!/usr/bin/env python3
"""Emit LaTeX cache tables (hit rate or timing) from the LIVE experiment results.

The measured counterpart of scripts/tog_cache_hitrate_table.py: instead of
simulating, it reads the real run summaries under artifacts/{rog,tog}_cache/
(the same files the plots use) and reports the *first-pass* numbers -- pass 1 for
ToG loop runs (from <tag>_pass1/summary.json), the single pass for RoG runs. That
is the cold-cache case, comparable across systems.

Two tables, chosen with --metric:

  hit_rate (default)  Rows are policies grouped into a RoG block and a ToG block;
                      columns are the live configurations (model x SPARQL
                      backend). Live runs are each a single cache capacity, so the
                      simulated table's capacity axis is replaced by that.

  timing              One table per configuration, since a second of RoG time and
                      a second of ToG time are only comparable within the same
                      model and backend. Rows stay (system, policy); columns are
                      hit rate, average miss/hit time, time saved, and the
                      full-system speedup.

The timing columns are *whole-question* times, which is the only unit in which the
two systems mean the same thing: rog_e2e_metrics.py joins RoG's stage-1 (planner)
and stage-2 (reasoner) sidecars per question and feeds them to ToG's own
aggregate_run_metrics(), so both systems' full_speedup_x come out of one function.
This deliberately is not a KG-time table -- RoG issues no KG requests to cache
(it walks the subgraph shipped in each HuggingFace example in memory), so KG time
exists only for ToG. That table is scripts/kg_timing_table.py.

    PYTHONPATH=src ./scripts/live_cache_hitrate_table.py
    PYTHONPATH=src ./scripts/live_cache_hitrate_table.py --metric timing
    PYTHONPATH=src ./scripts/live_cache_hitrate_table.py --dataset webqsp -o tables/live_hits.tex
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_plot():
    """Reuse the plot module's run discovery / summary loaders (import by path)."""
    path = REPO_ROOT / "scripts" / "plot_rog_cache_results.py"
    spec = importlib.util.spec_from_file_location("plot_rog_cache_results", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plot = _load_plot()

POLICY_LABELS = {
    "exact": "Exact Match",
    "semantic_lru": "Semantic LRU",
    "semantic_lfu": "Semantic LFU",
    "semantic_oracle": "Semantic Oracle",
}
DATASET_LABELS = {
    "webqsp": "WebQSP", "cwq": "CWQ", "qald": "QALD",
    "lcquad": "LC-QuAD", "lcquad_test": "LC-QuAD",
}
SYSTEM_LABELS = {"rog": "RoG", "tog": "ToG"}
DEFAULT_POLICIES = ["exact", "semantic_lru", "semantic_lfu", "semantic_oracle"]
# Column order for the live configurations (model, SPARQL backend).
VARIANT_ORDER = [("Gemini", "Oxi"), ("Gemini", "Virt"),
                 ("Haiku", "Oxi"), ("Haiku", "Virt")]


# A trailing power of two is the cache CAPACITY, not another configuration:
# `rog_live_virt_gemini_512` is the same run at a smaller cache. One table is one
# capacity, so the sweep variants stay out of it (scripts/plot_rog_cache_results.py
# is where the capacity axis belongs).
CAPACITY_TAG_RE = re.compile(r"_(?:64|128|256|512|1024|2048|4096)$")


def parse_tag(tag: str) -> tuple[str, str, str]:
    """(system, model, backend) parsed from a run tag; mirrors plot.pretty_label."""
    t = tag.lower()
    system = "tog" if "tog" in t else "rog"
    model = "Gemini" if "gemini" in t else "Haiku"
    backend = "Virt" if "virt" in t else "Oxi" if "oxi" in t else "?"
    return system, model, backend


def dataset_of(tag: str) -> str:
    """Dataset a run tag names. WebQSP is the unmarked default in this repo.

    Live tags carry the dataset only by convention -- `*_cwq*` for CWQ, nothing
    for WebQSP -- so this is what keeps a 3531-question CWQ run out of a WebQSP
    table. Before, --dataset set the caption and nothing else, and the two
    datasets' runs competed for the same column.
    """
    return "cwq" if "cwq" in tag.lower() else "webqsp"


def first_pass_records(system: str, tag: str) -> dict[str, dict]:
    """{policy: record} for the run's 1st pass: <tag>_pass1 if present, else whole."""
    p1 = plot._pass1_records(tag)      # only ToG loop runs have a _pass1 summary
    if p1 is not None:
        return p1
    base = plot.TOG_DIR if system == "tog" else plot.ROG_DIR
    return plot.load_run(base / tag / "summary.json")


def _fmt_hr(rec: dict | None, floor: int = 0) -> str:
    """Hit rate as a LaTeX percentage; '--' when the run/policy is missing.

    `floor` is the question count a policy must reach to be printed. A policy
    that ran a prefix of the split is not a measurement of that split -- the
    published version of this table drew 12.3% from 179 questions and 12.6% from
    494, against ~7.7% once each ran the full 1628 -- and a policy that ran
    nothing at all stores hit_rate 0.0, which prints as a hard `0.0%` and reads
    as a cache that never hit. Both become '--' here.
    """
    if not rec or not isinstance(rec.get("hit_rate"), (int, float)):
        return "--"
    if (rec.get("n_questions") or 0) < max(floor, 1):
        return "--"
    return f"{100 * rec['hit_rate']:.1f}\\%"


def time_saved_s(rec: dict) -> float | None:
    """Seconds the cache saved: what the hits would have cost as misses, minus
    what they did cost -- cache_metrics.aggregate_run_metrics()'s definition.

    Always recomputed, never read from the row's own `estimated_time_saved_s`.
    Two reasons, and both would corrupt the column: ToG's per-pass summaries do
    not write that key at all, and the RoG summaries on disk were written by an
    older formula that omitted the `- hit_total_s` term (e.g. 62 hits x 1.300 s =
    80.58 stored, against 72.98 net), so the stored numbers are gross avoided
    time. Recomputing gives every row in the table one meaning.

    None when there is no miss to price a hit against; 0.0 when a measured run
    simply got no hits.
    """
    hits, misses, hit_total = rec.get("hits"), rec.get("misses"), rec.get("hit_total_s")
    if not all(isinstance(v, (int, float)) for v in (hits, misses, hit_total)):
        return None
    if not misses:                       # no cold reference -> undefined
        return None
    if not hits:
        return 0.0
    # hit_total_s / miss_total_s cover only the questions that were actually
    # timed: a question whose LLM calls failed counts toward hits/misses (the
    # split it was scored against) but is kept out of the averages, so the
    # divisor for a total is timed_*, not hits/misses. Equal unless a run had
    # failures, and absent from summaries written before they were split.
    timed_hits = rec.get("timed_hits", hits)
    timed_misses = rec.get("timed_misses", misses)
    if not timed_misses:
        return None
    # From the totals rather than the rounded avg_miss_s, which is stored to 3dp
    # and is multiplied by the hit count here.
    miss_total = rec.get("miss_total_s")
    avg_miss = (miss_total / timed_misses if isinstance(miss_total, (int, float))
                else rec.get("avg_miss_s"))
    if not isinstance(avg_miss, (int, float)):
        return None
    return timed_hits * avg_miss - hit_total


def _fmt_secs(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "--"


def _fmt_speedup(value) -> str:
    return f"{value:.2f}$\\times$" if isinstance(value, (int, float)) else "--"


N_TIMING_CELLS = 6


def is_planner_only(system: str, recs: dict[str, dict]) -> bool:
    """True when a RoG run's timing covers stage 1 (planner) alone.

    summarize_rog_cache.py overwrites a row's bare timing fields with the
    whole-question numbers from rog_e2e_metrics.aggregate_end_to_end(), and stamps
    `e2e_metrics_file` when it does. A summary without that key was written before
    the stage-1 + stage-2 join existed, so its avg_hit_s / avg_miss_s /
    full_speedup_x describe the planner only -- excluding the reasoner's LLM call,
    the half of the pipeline the cache never touches. Those are an upper bound on
    the full-system effect and must not be printed beside ToG's whole-question
    times unmarked.

    ToG times a whole question inside one loop, so it is never stage-limited.
    """
    return system == "rog" and not any("e2e_metrics_file" in r for r in recs.values())


def timing_cells(rec: dict | None) -> list[str]:
    """The timing cells for one (system, policy); all '--' with no data.

    A policy whose sidecars never recorded a question (n_questions == 0) is an
    unmeasured run, not a zero -- its stored 0.0 timings would otherwise print as
    a real measurement of an instant cache.

    The question count leads the row on purpose: a resumed or interrupted run
    leaves each policy with however many questions it got through (these summaries
    range from 48 to 1628 within one configuration), and per-question averages
    from a 48-question policy do not carry the same weight as a full split's.
    """
    if not rec or not rec.get("n_questions"):
        return ["--"] * N_TIMING_CELLS
    return [
        str(rec["n_questions"]),
        _fmt_hr(rec),
        _fmt_secs(rec.get("avg_miss_s")),
        _fmt_secs(rec.get("avg_hit_s")),
        _fmt_secs(time_saved_s(rec)),
        _fmt_speedup(rec.get("full_speedup_x")),
    ]


def coverage(system: str, tag: str, policies: list[str]) -> tuple[int, int]:
    """(questions the THINNEST policy recorded, questions the fullest did).

    A run summary keeps whatever each policy got through, and an interrupted
    sweep leaves them wildly uneven -- `rog_cache_virtuoso_test` recorded 1628
    questions for LFU and 179 for LRU. The first number is what makes a run
    usable as a table column: every cell in it rests on the same questions only
    when the thinnest policy is as deep as the rest.
    """
    try:
        recs = first_pass_records(system, tag)
    except (OSError, ValueError):
        return 0, 0
    counts = [recs.get(p, {}).get("n_questions") or 0 for p in policies]
    return (min(counts), max(counts)) if counts else (0, 0)


def discover_runs(dataset: str, policies: list[str]) -> dict[tuple[str, str, str], str]:
    """{(system, model, backend): tag}: the best run per column, for one dataset.

    Best = the run whose thinnest requested policy is deepest, then the fullest,
    then the latest tag. Picking on coverage rather than on tag order is the
    point: the old rule kept whichever tag sorted last and so handed the RoG
    Gemini/Virtuoso column to a capacity-sweep run with two empty policies, while
    a complete 1628-question run sat next to it.
    """
    best: dict[tuple[str, str, str], tuple[int, int, str]] = {}
    for base in (plot.ROG_DIR, plot.TOG_DIR):
        for tag in plot.glob_runs(base):          # skips GPT and per-pass summaries
            if dataset_of(tag) != dataset or CAPACITY_TAG_RE.search(tag):
                continue
            key = parse_tag(tag)
            rank = (*coverage(key[0], tag, policies), tag)
            if key not in best or rank > best[key]:
                best[key] = rank
    return {k: v[2] for k, v in best.items()}


def build_table(dataset: str, systems: list[str], variants: list[tuple[str, str]],
                policies: list[str],
                cells: dict[tuple[str, str, tuple[str, str]], str],
                counts: dict[tuple[str, tuple[str, str]], int] | None = None) -> str:
    """Render the LaTeX table: System | Policy | one column per (model, backend).

    The caption carries each column's question count. Percentages alone hide the
    denominator, and the columns of this table are not obliged to share one: on
    CWQ the RoG run covers 3531 questions and the ToG run 400, which a reader
    comparing 2.4% against 2.5% has to be told.
    """
    colspec = "ll" + "r" * len(variants)
    dlabel = DATASET_LABELS.get(dataset.lower(), dataset.upper())
    caption = (f"Measured First-Pass Semantic Cache Hit Rate by Policy and "
               f"Configuration on {dlabel} (live runs)")
    if counts:
        ns = "; ".join(f"{SYSTEM_LABELS[sy]} {m} ({b}) {counts[(sy, (m, b))]}"
                       for sy in systems for m, b in variants
                       if (sy, (m, b)) in counts)
        if ns:
            caption += f". Questions recorded per run: {ns}"
    col_heads = " & ".join(rf"\textbf{{{m} ({b})}}" for m, b in variants)
    header = rf"\textbf{{System}} & \textbf{{Policy}} & {col_heads} \\"

    rows: list[str] = []
    for si, system in enumerate(systems):
        for pi, p in enumerate(policies):
            sys_cell = f"{SYSTEM_LABELS[system]:<4}" if pi == 0 else f"{'':<4}"
            vals = " & ".join(cells.get((system, p, v), "--") for v in variants)
            rows.append(f"{sys_cell} & {POLICY_LABELS.get(p, p):<16} & {vals} \\\\")
        if si != len(systems) - 1:
            rows.append(r"\hline")

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:live_cache_hits_{dataset.lower()}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        header,
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])


def build_timing_table(dataset: str, systems: list[str], variant: tuple[str, str],
                       policies: list[str],
                       cells: dict[tuple[str, str], list[str]],
                       stage1_only: set[str] | None = None) -> str:
    """Render one configuration's timing table: System | Policy | the metrics.

    Systems in `stage1_only` get a dagger and a caption note: their times cover
    the planner stage alone, so they are not on ToG's whole-question scale.
    """
    stage1_only = stage1_only or set()
    model, backend = variant
    dlabel = DATASET_LABELS.get(dataset.lower(), dataset.upper())
    scale = ("Times are whole-question seconds (planner + reasoner), so RoG and "
             "ToG are on the same scale.")
    if stage1_only:
        marked = " and ".join(SYSTEM_LABELS[s] for s in systems if s in stage1_only)
        whole = [SYSTEM_LABELS[s] for s in systems if s not in stage1_only]
        lead = (f"{' and '.join(whole)} times are whole-question seconds "
                f"(planner + reasoner). " if whole else "")
        scale = (lead + f"$\\dagger$ {marked} times cover the planner stage only: "
                 "those run summaries predate the stage-1 + stage-2 join, so they "
                 "exclude the reasoner and are an upper bound on the cache's "
                 "full-system effect. Re-run summarize\\_rog\\_cache.py to record "
                 "end-to-end times.")
    caption = (f"Measured First-Pass Cache Timing by Policy on {dlabel}, "
               f"{model} ({backend}). {scale} "
               f"$N$ is the number of questions the policy actually recorded.")
    header = (r"\textbf{System} & \textbf{Policy} & \textbf{$N$} & \textbf{Hit Rate} & "
              r"\textbf{Avg Miss (s)} & \textbf{Avg Hit (s)} & \textbf{Saved (s)} & "
              r"\textbf{Full Speedup} \\")

    rows: list[str] = []
    for si, system in enumerate(systems):
        name = SYSTEM_LABELS[system]
        if system in stage1_only:
            name += r"$^{\dagger}$"
        for pi, p in enumerate(policies):
            sys_cell = f"{name:<4}" if pi == 0 else f"{'':<4}"
            vals = " & ".join(cells.get((system, p), ["--"] * N_TIMING_CELLS))
            rows.append(f"{sys_cell} & {POLICY_LABELS.get(p, p):<16} & {vals} \\\\")
        if si != len(systems) - 1:
            rows.append(r"\hline")

    slug = f"{dataset.lower()}_{model.lower()}_{backend.lower()}"
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:live_cache_timing_{slug}}}",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        header,
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="webqsp",
                    help="dataset name for the caption/label (live tags don't encode it)")
    ap.add_argument("-m", "--metric", default="hit_rate", choices=["hit_rate", "timing"],
                    help="hit_rate: policies x configurations, one table. "
                         "timing: one table per configuration, with the timing columns")
    ap.add_argument("-p", "--policies", default=",".join(DEFAULT_POLICIES),
                    help="comma-separated policies (rows)")
    ap.add_argument("-s", "--systems", default="rog,tog",
                    help="comma-separated systems: rog, tog")
    ap.add_argument("-r", "--run", action="append", default=[], metavar="SPEC",
                    help="pin a column to a run tag instead of discovering it, as "
                         "system:model:backend=tag (e.g. "
                         "-r tog:Haiku:Oxi=tog_rerun_live_oxi). Repeatable.")
    ap.add_argument("--min-coverage", type=float, default=0.9, metavar="FRAC",
                    help="a policy must have recorded this fraction of its run's "
                         "fullest policy to be printed, else '--' (default: 0.9). "
                         "Stops a partial policy sweep from reading as a "
                         "measurement of the split. Pass 0 to print everything.")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write the .tex here (default: stdout)")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in SYSTEM_LABELS:
            sys.exit(f"unknown system: {s!r} (choose from {tuple(SYSTEM_LABELS)})")

    runs = discover_runs(args.dataset.lower(), policies)
    for spec in args.run:
        key, _, tag = spec.partition("=")
        parts = [x.strip() for x in key.split(":")]
        if not tag or len(parts) != 3:
            sys.exit(f"bad --run {spec!r}; expected system:model:backend=tag")
        system, model, backend = parts[0].lower(), parts[1], parts[2]
        if system not in SYSTEM_LABELS:
            sys.exit(f"bad --run {spec!r}: unknown system {system!r}")
        base = plot.ROG_DIR if system == "rog" else plot.TOG_DIR
        if not (base / tag / "summary.json").exists():
            sys.exit(f"bad --run {spec!r}: no summary at {base / tag / 'summary.json'}")
        runs[(system, model, backend)] = tag
    if not runs:
        sys.exit(f"no {args.dataset} run summaries under artifacts/{{rog,tog}}_cache/ "
                 f"(tags are matched by name: '*cwq*' is CWQ, anything else WebQSP)")

    # Only keep configuration columns that actually exist for some requested system.
    variants = [v for v in VARIANT_ORDER
                if any((s, v[0], v[1]) in runs for s in systems)]

    # {(system, variant): {policy: record}}, loaded once and shared by both tables.
    loaded: dict[tuple[str, tuple[str, str]], dict[str, dict]] = {}
    for system in systems:
        for variant in variants:
            tag = runs.get((system, variant[0], variant[1]))
            if tag is None:
                print(f"[live] no run for {SYSTEM_LABELS[system]} {variant[0]} ({variant[1]})",
                      file=sys.stderr)
                continue
            loaded[(system, variant)] = first_pass_records(system, tag)
            lo, hi = coverage(system, tag, policies)
            note = "" if lo == hi else f" (thinnest policy {lo} q)"
            print(f"[live] {SYSTEM_LABELS[system]} {variant[0]} ({variant[1]}) <- {tag} "
                  f"[{hi} q{note}]", file=sys.stderr)

    if args.metric == "timing":
        # One table per configuration: seconds are only comparable within a fixed
        # model and backend, so a column spanning configurations would mislead.
        tables = []
        for variant in variants:
            cells = {(system, p): timing_cells(loaded.get((system, variant), {}).get(p))
                     for system in systems for p in policies}
            stage1_only = {s for s in systems
                           if is_planner_only(s, loaded.get((s, variant), {}))}
            for s in stage1_only:
                print(f"[live] {SYSTEM_LABELS[s]} {variant[0]} ({variant[1]}): planner-stage "
                      f"timing only (no e2e_metrics_file in the summary)", file=sys.stderr)
            tables.append(build_timing_table(args.dataset, systems, variant, policies,
                                             cells, stage1_only))
        tex = "\n\n".join(tables) + "\n"
    else:
        cells = {}
        for system in systems:
            for variant in variants:
                recs = loaded.get((system, variant), {})
                floor = round(args.min_coverage * coverage(
                    system, runs.get((system, *variant), ""), policies)[1])
                for p in policies:
                    rec = recs.get(p)
                    cells[(system, p, variant)] = _fmt_hr(rec, floor)
                    if rec and cells[(system, p, variant)] == "--" and floor:
                        print(f"[live] {SYSTEM_LABELS[system]} {variant[0]} "
                              f"({variant[1]}) {p}: {rec.get('n_questions') or 0} of "
                              f"{floor} questions needed -- dropped as partial",
                              file=sys.stderr)
        counts = {(sy, v): coverage(sy, runs[(sy, *v)], policies)[1]
                  for sy in systems for v in variants if (sy, *v) in runs}
        tex = build_table(args.dataset, systems, variants, policies,
                          cells, counts) + "\n"
    if args.output:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(f"[live] wrote {out}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
