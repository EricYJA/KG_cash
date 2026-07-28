#!/usr/bin/env python3
"""Emit a LaTeX cache-hit-rate table from the LIVE experiment results.

The measured counterpart of scripts/tog_cache_hitrate_table.py: instead of
simulating, it reads the real run summaries under artifacts/{rog,tog}_cache/
(the same files the plots use) and reports the *first-pass* cache hit rate --
pass 1 for ToG loop runs (from <tag>_pass1/summary.json), the single pass for
RoG runs. That is the cold-cache rate, comparable across systems.

Live runs are each a single cache capacity, so the capacity axis of the simulated
table has no meaning here; the columns instead become the live configurations
(model x SPARQL backend) and the rows stay the cache policies, grouped into a RoG
block and a ToG block -- the same LaTeX shape as the simulated table.

    PYTHONPATH=src ./scripts/live_cache_hitrate_table.py
    PYTHONPATH=src ./scripts/live_cache_hitrate_table.py --dataset webqsp -o tables/live_hits.tex
"""
from __future__ import annotations

import argparse
import importlib.util
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


def parse_tag(tag: str) -> tuple[str, str, str]:
    """(system, model, backend) parsed from a run tag; mirrors plot.pretty_label."""
    t = tag.lower()
    system = "tog" if "tog" in t else "rog"
    model = "Gemini" if "gemini" in t else "Haiku"
    backend = "Virt" if "virtuoso" in t else "Oxi" if "oxi" in t else "?"
    return system, model, backend


def first_pass_records(system: str, tag: str) -> dict[str, dict]:
    """{policy: record} for the run's 1st pass: <tag>_pass1 if present, else whole."""
    p1 = plot._pass1_records(tag)      # only ToG loop runs have a _pass1 summary
    if p1 is not None:
        return p1
    base = plot.TOG_DIR if system == "tog" else plot.ROG_DIR
    return plot.load_run(base / tag / "summary.json")


def _fmt_hr(rec: dict | None) -> str:
    """Hit rate as a LaTeX percentage; '--' when the run/policy is missing."""
    if not rec or not isinstance(rec.get("hit_rate"), (int, float)):
        return "--"
    return f"{100 * rec['hit_rate']:.1f}\\%"


def discover_runs() -> dict[tuple[str, str, str], str]:
    """{(system, model, backend): tag} for every whole-run summary on disk."""
    runs: dict[tuple[str, str, str], str] = {}
    for base in (plot.ROG_DIR, plot.TOG_DIR):
        for tag in plot.glob_runs(base):          # skips GPT and per-pass summaries
            runs[parse_tag(tag)] = tag
    return runs


def build_table(dataset: str, systems: list[str], variants: list[tuple[str, str]],
                policies: list[str],
                cells: dict[tuple[str, str, tuple[str, str]], str]) -> str:
    """Render the LaTeX table: System | Policy | one column per (model, backend)."""
    colspec = "ll" + "r" * len(variants)
    dlabel = DATASET_LABELS.get(dataset.lower(), dataset.upper())
    caption = (f"Measured First-Pass Semantic Cache Hit Rate by Policy and "
               f"Configuration on {dlabel} (live runs)")
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="webqsp",
                    help="dataset name for the caption/label (live tags don't encode it)")
    ap.add_argument("-p", "--policies", default=",".join(DEFAULT_POLICIES),
                    help="comma-separated policies (rows)")
    ap.add_argument("-s", "--systems", default="rog,tog",
                    help="comma-separated systems: rog, tog")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write the .tex here (default: stdout)")
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    for s in systems:
        if s not in SYSTEM_LABELS:
            sys.exit(f"unknown system: {s!r} (choose from {tuple(SYSTEM_LABELS)})")

    runs = discover_runs()
    if not runs:
        sys.exit("no live run summaries found under artifacts/{rog,tog}_cache/")

    # Only keep configuration columns that actually exist for some requested system.
    variants = [v for v in VARIANT_ORDER
                if any((s, v[0], v[1]) in runs for s in systems)]

    cells: dict[tuple[str, str, tuple[str, str]], str] = {}
    for system in systems:
        for (model, backend) in variants:
            tag = runs.get((system, model, backend))
            if tag is None:
                print(f"[live] no run for {SYSTEM_LABELS[system]} {model} ({backend})",
                      file=sys.stderr)
                continue
            recs = first_pass_records(system, tag)
            for p in policies:
                cells[(system, p, (model, backend))] = _fmt_hr(recs.get(p))
            print(f"[live] {SYSTEM_LABELS[system]} {model} ({backend}) <- {tag}",
                  file=sys.stderr)

    tex = build_table(args.dataset, systems, variants, policies, cells) + "\n"
    if args.output:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(f"[live] wrote {out}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
