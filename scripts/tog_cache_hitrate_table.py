#!/usr/bin/env python3
"""Emit LaTeX hit-rate tables (RoG + ToG, policy x cache capacity), 1st pass only.

Reuses the fast cache simulator (src/ToG-cache/ToG/simulate_cache.py) -- no LLM,
no SPARQL, no GPU beyond the sentence-embedder -- to sweep every (policy, cache
capacity) pair for both retrieval paradigms and print one LaTeX table per
configuration, e.g. "Semantic Cache Hit Rate by Policy and Cache Capacity on
WebQSP for Threshold 0.9", with a RoG block and a ToG block.

Both systems share the same cache mechanics (the RoG simulator explicitly reuses
ToG's FastSimCache); they differ only in their question stream -- ToG walks its
local `prepare_dataset`, RoG the HuggingFace `rmanluo/RoG-<dataset>` split -- so
the hit rates can differ even on the "same" benchmark. Oracle keys for both come
from the one extract_oracle_answer_key (it has an `rog-*` branch), so the
semantic_oracle policy means the same thing in each block.

"1st pass only" == the simulator run with a single pass: each question is looked
up once, so the reported rate is the cold-cache (first-encounter) rate -- the same
number the pass-1 plots show, not the warm-pass-inflated aggregate.

One table per (dataset, threshold); pass several of either to sweep them
(embeddings are computed once per system+dataset and reused across thresholds).
Progress goes to stderr, so stdout stays clean LaTeX.

Examples:
    # default: RoG+ToG, WebQSP, threshold 0.9, capacities 32/128/512/2048/inf
    PYTHONPATH=src ./scripts/tog_cache_hitrate_table.py

    # several configs to a file; ToG only
    PYTHONPATH=src ./scripts/tog_cache_hitrate_table.py -d webqsp,cwq -t 0.85,0.90 \
        --systems tog -o tables/cache_hits.tex
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOG_DIR = REPO_ROOT / "src" / "ToG-cache" / "ToG"


def _load_sim():
    """Import simulate_cache.py by path (ToG/ is not a package)."""
    path = TOG_DIR / "simulate_cache.py"
    spec = importlib.util.spec_from_file_location("simulate_cache", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim = _load_sim()

# Display names matching the target table.
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
# Canonical display name per system key.
SYSTEM_LABELS = {"rog": "RoG", "tog": "ToG"}
ROG_SPLIT = "test"


def _cap_header(c: int) -> str:
    """Capacity column header; the sentinel 10**9 renders as an infinity symbol."""
    return r"$\infty$" if c >= 10**9 else str(c)


def _fmt_hr(hr: float) -> str:
    """Hit rate as a LaTeX percentage, one decimal (e.g. 8.2\\%)."""
    return f"{100 * hr:.1f}\\%"


def _dataset_label(d: str) -> str:
    return DATASET_LABELS.get(d.lower(), d.upper())


def _label(dataset: str, threshold: float, systems: list[str]) -> str:
    sys_part = "".join(systems)
    return f"tab:cache_hits_{sys_part}_{dataset.lower()}_t{f'{threshold:g}'.replace('.', '')}"


def load_source(system: str, dataset: str, limit, need_oracle: bool):
    """(datas, question_key, oracle_keys) for one system's question stream.

    ToG walks its local prepare_dataset; RoG the HuggingFace rmanluo/RoG-<dataset>
    split. Both are fed to ToG's simulate(), which only reads data[question_key]
    and the parallel oracle_keys list.
    """
    if system == "tog":
        datas, qstr = sim.load_dataset(dataset, limit)
        oracle_keys = ([sim.extract_oracle_answer_key(d, dataset) for d in datas]
                       if need_oracle else None)
        return datas, qstr, oracle_keys

    # RoG: HuggingFace dataset; oracle keys use the shared extractor's rog-* branch.
    from datasets import load_dataset as hf_load
    ds = hf_load(f"rmanluo/RoG-{dataset}", split=ROG_SPLIT)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    datas = [{"question": d["question"]} for d in ds]
    oracle_keys = ([sim.extract_oracle_answer_key(d, f"rog-{dataset}") for d in ds]
                   if need_oracle else None)
    return datas, "question", oracle_keys


def build_table(dataset: str, threshold: float, systems: list[str],
                capacities: list[int], policies: list[str],
                results: dict[tuple[str, str, int], float]) -> str:
    """Render one LaTeX table; a leading System column groups the blocks when >1."""
    two_systems = len(systems) > 1
    colspec = ("ll" if two_systems else "l") + "r" * len(capacities)
    caption = (f"Semantic Cache Hit Rate by Policy and Cache Capacity on "
               f"{_dataset_label(dataset)} for Threshold {threshold:g}")

    sys_head = r"\textbf{System} & " if two_systems else ""
    header = (sys_head + r"\textbf{Policy} & "
              + " & ".join(rf"\textbf{{{_cap_header(c)}}}" for c in capacities)
              + r" \\")

    rows: list[str] = []
    for si, system in enumerate(systems):
        for pi, p in enumerate(policies):
            cells = " & ".join(_fmt_hr(results[(system, p, c)]) for c in capacities)
            label = f"{POLICY_LABELS.get(p, p):<16}"
            if two_systems:
                # System name only on the block's first row; blank thereafter.
                sys_cell = f"{SYSTEM_LABELS[system]:<4}" if pi == 0 else f"{'':<4}"
                rows.append(f"{sys_cell} & {label} & {cells} \\\\")
            else:
                rows.append(f"{label} & {cells} \\\\")
        if two_systems and si != len(systems) - 1:
            rows.append(r"\hline")

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{_label(dataset, threshold, systems)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        header,
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])


def run_dataset(dataset: str, systems: list[str], thresholds: list[float],
                capacities: list[int], policies: list[str], limit,
                embedder_model: str) -> list[str]:
    """Simulate every (system, policy, capacity, threshold) -> LaTeX tables."""
    need_oracle = "semantic_oracle" in policies
    uses_embed = any(p in sim._USES_EMBEDDING for p in policies)

    # Load each system's question stream once; embeddings depend only on the
    # questions, so reuse them across this dataset's thresholds.
    prepared: dict[str, tuple] = {}
    for system in systems:
        datas, qstr, oracle_keys = load_source(system, dataset, limit, need_oracle)
        print(f"[table] {SYSTEM_LABELS[system]}: {len(datas)} questions from {dataset!r}",
              file=sys.stderr)
        embed_map = sim.precompute_embeddings(datas, qstr, embedder_model) if uses_embed else None
        prepared[system] = (datas, qstr, oracle_keys, embed_map)

    tables: list[str] = []
    for threshold in thresholds:
        results: dict[tuple[str, str, int], float] = {}
        for system in systems:
            datas, qstr, oracle_keys, embed_map = prepared[system]
            for p in policies:
                for c in capacities:
                    stats = sim.simulate(datas, qstr, p, c, threshold, embedder_model,
                                         1, oracle_keys, embed_map)  # 1 pass = 1st pass
                    results[(system, p, c)] = stats["hit_rate"]
                    print(f"  [{SYSTEM_LABELS[system]} {dataset} t={threshold:g}] "
                          f"{p:<16} cap={_cap_header(c):<8} -> {_fmt_hr(results[(system, p, c)])}",
                          file=sys.stderr)
        tables.append(build_table(dataset, threshold, systems, capacities, policies, results))
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--datasets", default="webqsp",
                    help="comma-separated datasets (one table per dataset x threshold)")
    ap.add_argument("-t", "--thresholds", default="0.90",
                    help="comma-separated cosine thresholds")
    ap.add_argument("-c", "--capacities", default="32,128,512,2048,inf",
                    help="comma-separated cache capacities; 'inf' for unbounded")
    ap.add_argument("-p", "--policies",
                    default="exact,semantic_lru,semantic_lfu,semantic_oracle",
                    help=f"comma-separated policies; choose from {sim.POLICIES}")
    ap.add_argument("-s", "--systems", default="rog,tog",
                    help="comma-separated systems: rog, tog (default: both)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap the number of questions (default: whole dataset)")
    ap.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write the .tex here (default: stdout)")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    capacities = sim.parse_capacity_list(args.capacities)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    for p in policies:
        if p not in sim.POLICIES:
            sys.exit(f"unknown policy: {p!r} (choose from {sim.POLICIES})")
    for s in systems:
        if s not in SYSTEM_LABELS:
            sys.exit(f"unknown system: {s!r} (choose from {tuple(SYSTEM_LABELS)})")

    # Resolve the output path up front (load_dataset chdirs then restores, but be safe).
    out_path = args.output.resolve() if args.output else None

    tables: list[str] = []
    for dataset in datasets:
        tables += run_dataset(dataset, systems, thresholds, capacities, policies,
                              args.limit, args.embedder_model)

    tex = "\n\n".join(tables) + "\n"
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tex)
        print(f"[table] wrote {len(tables)} table(s) to {out_path}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
