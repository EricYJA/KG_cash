#!/usr/bin/env python3
"""Emit a LaTeX KG-retrieval-time table from the simulated cache results.

The table counterpart of src/ToG-cache/ToG/plot_cache_time_breakdown.py: both read
the combined simulation summary written by cache_simulator.py --combined, but this
prints the KG timing as numbers instead of a stacked-bar figure. Per (dataset,
policy) it reports the cold KG time, the KG time with the cache, the seconds saved,
and the resulting KG speedup -- the quantities the breakdown figure encodes as the
zoomed row and its twin-axis speedup line.

Source of the numbers, and why they are trustworthy: cache_simulator.py replays a
recorded ToG trace (output/traces/tog_trace_<dataset>.json). Every SPARQL call ToG
made is one KG event in that trace with a cache key and a measured duration_ms, so
"KG time with a cache" is the sum of the durations of just the requests that would
have missed. The LLM time is untouched by the cache and is left out of this table
entirely -- it dwarfs the KG time (11477 s vs 57 s on WebQSP), so a total-time
column would hide the whole effect.

The cached stage differs by system, and this is a property of the two systems
rather than missing plumbing. ToG retrieves over the network (freebase_func.py ->
SPARQL_ENDPOINT), so its KG requests are individually timed and cacheable. RoG
never issues a KG request in this repo: it grounds relation paths with
utils.build_graph(sample["graph"]) plus bfs_with_rule, an in-memory networkx walk
over the subgraph shipped inside each HuggingFace example. There is no KG request
stream to replay. What RoG's cache does elide is the per-question planner LLM
call, so src/RoG-cache/simulate_rog_cache_timing.py replays *that* instead, from
the per-question sidecars of an uncached run. It writes the same summary shape,
and this script reads either one -- metadata.stage_keys names which time bucket
the cache shrinks ("kg" for ToG, "planner" for RoG).

    # default: every dataset and policy, largest capacity, sequential access
    ./scripts/kg_timing_table.py

    # sweep capacities, write to a file
    ./scripts/kg_timing_table.py -c 100,500,1000 -o tables/kg_timing.tex

    # the RoG planner-time table
    ./scripts/kg_timing_table.py -i artifacts/rog_cache_sim/rog_cache_sim_summary.json

Regenerate the inputs first if the traces changed:
    python src/ToG-cache/ToG/cache_simulator.py --combined \
        --cache-sizes 10,50,100,500,1000 --policies lru,lfu,fifo,random,belady,oracle
    python src/RoG-cache/simulate_rog_cache_timing.py \
        --run artifacts/rog_cache/rog_cache_virtuoso_new
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "src" / "ToG-cache" / "output" / "cache_sim_summary.json"

POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "fifo": "FIFO",
    "random": "Random",
    "belady": "Belady (MIN)",
    "oracle": "Oracle (preload)",
    "exact": "Exact",
    "semantic_lru": "LRU",
    "semantic_lfu": "LFU",
    "semantic_fifo": "FIFO",
    "semantic_random": "Random",
    "semantic_belady": "Belady (MIN)",
    "semantic_oracle": "Oracle (gold-gated)",
}
# Preferred row order; anything unrecognised is appended after.
POLICY_ORDER = [
    "exact", "lru", "semantic_lru", "lfu", "semantic_lfu",
    "fifo", "semantic_fifo", "random", "semantic_random",
    "belady", "semantic_belady", "oracle", "semantic_oracle",
]
# The simulator keys its datasets by display name already ("WebQSP", "CWQ"), so
# this only normalizes what the user types on the command line.
DATASET_ALIASES = {"webqsp": "WebQSP", "cwq": "CWQ"}
ACCESS_LABELS = {"sequential": "sequential", "shuffled": "shuffled"}
# Older summaries predate metadata.stage_keys; they were all ToG KG/LLM runs.
DEFAULT_STAGE_KEYS = {"cached": "kg", "uncached": "llm"}
DEFAULT_STAGE_LABELS = {"cached": "KG", "uncached": "LLM"}


def stage_config(payload: dict) -> tuple[dict, dict]:
    """Which time bucket the cache shrinks ("kg" for ToG, "planner" for RoG)."""
    metadata = payload.get("metadata", {})
    keys = {**DEFAULT_STAGE_KEYS, **metadata.get("stage_keys", {})}
    labels = {**DEFAULT_STAGE_LABELS, **metadata.get("stage_labels", {})}
    return keys, labels


def _ms_to_s(ms: float) -> float:
    return ms / 1000.0


def _fmt_hr(hits: int, requests: int) -> str:
    return f"{100 * hits / requests:.1f}\\%" if requests else "--"


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.1f}"


def _fmt_speedup(base_ms: float, cached_ms: float) -> str:
    """KG speedup as base/cached; '--' when the cache eliminated all KG time."""
    return f"{base_ms / cached_ms:.2f}$\\times$" if cached_ms > 0 else "--"


def load_payload(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"no simulation summary at {path}\n"
                 f"generate it with: python src/ToG-cache/ToG/cache_simulator.py --combined")
    payload = json.loads(path.read_text())
    if "datasets" not in payload:
        sys.exit(f"not a combined cache summary (no 'datasets' key): {path}")
    return payload


def find_record(records: list[dict], policy: str, cache_size: int) -> dict | None:
    for record in records:
        if record["policy"] == policy and record["cache_size"] == cache_size:
            return record
    return None


def build_hit_rate_table(payload: dict, datasets: list[str], policies: list[str],
                         cache_sizes: list[int], access: str) -> str:
    """Hit rate as policies x cache sizes -- the comparison figure, as numbers.

    This is the only table that says anything for a summary with no measured
    durations (RoG walks its KG in memory, so its time columns are all zero).
    """
    colspec = "ll" + "r" * len(cache_sizes)
    header = (r"\textbf{Dataset} & \textbf{Policy} & "
              + " & ".join(rf"\textbf{{{size}}}" for size in cache_sizes) + r" \\")
    caption = (f"Simulated cache hit rate by policy and cache size "
               f"({ACCESS_LABELS[access]} access). Columns are cache capacities.")
    label = f"tab:hit_rate_{access}"

    rows: list[str] = []
    for index, dataset in enumerate(datasets):
        records = payload["datasets"][dataset][access]
        first = True
        for policy in policies:
            cells = []
            for cache_size in cache_sizes:
                rec = find_record(records, policy, cache_size)
                cells.append(_fmt_hr(rec["hits"], rec["requests"]) if rec else "--")
            ds_cell = f"{dataset:<10}" if first else f"{'':<10}"
            first = False
            rows.append(f"{ds_cell} & {POLICY_LABELS.get(policy, policy):<18} "
                        f"& {' & '.join(cells)} \\\\")
        # The reuse ceiling is what makes a preload 'oracle' above it legible as
        # an artifact rather than a result, so it is stated with the data.
        info = payload["datasets"][dataset]
        total, unique = info.get("request_count"), info.get("unique_entities")
        if total and unique:
            rows.append(rf"\multicolumn{{{2 + len(cache_sizes)}}}{{l}}{{\footnotesize "
                        rf"{dataset}: {total} requests, {unique} unique "
                        rf"-- reuse ceiling {100 * (1 - unique / total):.1f}\%}} \\")
        if index != len(datasets) - 1:
            rows.append(r"\hline")

    return "\n".join([
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\hline", header, r"\hline",
        *rows, r"\hline", r"\end{tabular}", r"\end{table}",
    ])


def build_table(payload: dict, datasets: list[str], policies: list[str],
                cache_sizes: list[int], access: str) -> str:
    """Render the LaTeX table: one block per dataset, one row per (capacity, policy)."""
    stage_keys, stage_labels = stage_config(payload)
    cached_key, uncached_key = stage_keys["cached"], stage_keys["uncached"]
    stage = stage_labels["cached"]
    # The capacity column only earns its place when more than one was requested;
    # with a single capacity it is constant and moves into the caption instead.
    show_cap = len(cache_sizes) > 1
    colspec = ("lll" if show_cap else "ll") + "r" * 5
    cap_head = r"\textbf{Capacity} & " if show_cap else ""
    header = (r"\textbf{Dataset} & " + cap_head + r"\textbf{Policy} & "
              r"\textbf{Hit Rate} & " + rf"\textbf{{{stage} (no cache)}} & "
              rf"\textbf{{{stage} (cached)}} & "
              rf"\textbf{{Saved}} & \textbf{{{stage} Speedup}} \\")

    cap_note = "" if show_cap else f", cache capacity {cache_sizes[0]}"
    caption = (f"Simulated {stage} Time and Speedup by Cache Policy "
               f"({ACCESS_LABELS[access]} access{cap_note}). Times are seconds of {stage} "
               f"time over the whole trace; {stage_labels['uncached']} time is unaffected "
               f"by the cache and is excluded.")
    cap_label = "sweep" if show_cap else str(cache_sizes[0])
    # Stage-qualified so the ToG (kg) and RoG (planner) tables can sit in the
    # same document without clashing on \label.
    label = f"tab:{cached_key}_timing_{access}_{cap_label}"

    rows: list[str] = []
    for di, dataset in enumerate(datasets):
        records = payload["datasets"][dataset][access]
        first_row_of_block = True
        for cache_size in cache_sizes:
            for policy in policies:
                rec = find_record(records, policy, cache_size)
                if rec is None:
                    print(f"[kg-timing] no record for {dataset} {policy} cap={cache_size}",
                          file=sys.stderr)
                    continue
                timing = rec["time_breakdown_ms"]
                base_ms = timing[f"{cached_key}_base"]
                cached_ms = timing[f"{cached_key}_simulated"]
                cells = [
                    _fmt_hr(rec["hits"], rec["requests"]),
                    _fmt_s(_ms_to_s(base_ms)),
                    _fmt_s(_ms_to_s(cached_ms)),
                    _fmt_s(_ms_to_s(base_ms - cached_ms)),
                    _fmt_speedup(base_ms, cached_ms),
                ]
                # Dataset name only on the block's first row; blank thereafter.
                ds_cell = f"{dataset:<8}" if first_row_of_block else f"{'':<8}"
                first_row_of_block = False
                cap_cell = f"{cache_size:<8} & " if show_cap else ""
                rows.append(f"{ds_cell} & {cap_cell}{POLICY_LABELS.get(policy, policy):<8}"
                            f" & {' & '.join(cells)} \\\\")
        if di != len(datasets) - 1:
            rows.append(r"\hline")

    if not rows:
        sys.exit("no matching records: check --datasets/--policies/--cache-sizes")

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
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
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT,
                    help="combined cache simulation JSON (cache_simulator.py --combined)")
    ap.add_argument("-d", "--datasets", default="",
                    help="comma-separated datasets (default: every dataset in the input)")
    ap.add_argument("-p", "--policies", default="",
                    help="comma-separated policies (rows); default: every policy in the input")
    ap.add_argument("-c", "--cache-sizes", default="",
                    help="comma-separated capacities (default: the largest in the input, "
                         "which is the operating point the breakdown figure plots)")
    ap.add_argument("-m", "--metric", default="timing", choices=["timing", "hit-rate"],
                    help="'timing' needs measured durations (ToG); 'hit-rate' is the "
                         "comparison figure as a table and works for any summary")
    ap.add_argument("--access", default="sequential", choices=["sequential", "shuffled"],
                    help="access pattern to report (default: sequential = the real order)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write the .tex here (default: stdout)")
    args = ap.parse_args()

    payload = load_payload(args.input)
    available = list(payload["datasets"])
    meta = payload.get("metadata", {})

    if args.datasets.strip():
        datasets = []
        for d in (x.strip() for x in args.datasets.split(",") if x.strip()):
            name = DATASET_ALIASES.get(d.lower(), d)
            if name not in payload["datasets"]:
                sys.exit(f"unknown dataset: {d!r} (input has {available})")
            datasets.append(name)
    else:
        datasets = available

    sim_policies = meta.get("policies", [])
    if args.policies.strip():
        policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
        for p in policies:
            if sim_policies and p not in sim_policies:
                sys.exit(f"policy {p!r} was not simulated (input has {sim_policies})")
    else:
        # ToG and RoG use different policy vocabularies ("lru" vs "semantic_lru"),
        # so take whatever the input actually holds rather than a fixed default.
        if not sim_policies:
            sys.exit("input has no metadata.policies; pass --policies explicitly")
        ranked = [p for p in POLICY_ORDER if p in sim_policies]
        policies = ranked + [p for p in sim_policies if p not in ranked]

    sim_sizes = meta.get("cache_sizes", [])
    if args.cache_sizes.strip():
        cache_sizes = [int(c.strip()) for c in args.cache_sizes.split(",") if c.strip()]
        for c in cache_sizes:
            if sim_sizes and c not in sim_sizes:
                sys.exit(f"capacity {c} was not simulated (input has {sim_sizes})")
    else:
        if not sim_sizes:
            sys.exit("input has no metadata.cache_sizes; pass --cache-sizes explicitly")
        # A hit-rate table is a sweep by nature; a timing table reports one
        # operating point unless asked otherwise.
        cache_sizes = sim_sizes if args.metric == "hit-rate" else [max(sim_sizes)]

    if args.metric == "hit-rate":
        tex = build_hit_rate_table(payload, datasets, policies, cache_sizes, args.access) + "\n"
    else:
        stage_keys, _ = stage_config(payload)
        sample = payload["datasets"][datasets[0]][args.access][0]["time_breakdown_ms"]
        if not sample.get(f"{stage_keys['cached']}_base"):
            print(f"[kg-timing] warning: this summary has no measured durations "
                  f"(every {stage_keys['cached']} time is 0), so the timing columns will be "
                  f"empty. Use --metric hit-rate.", file=sys.stderr)
        tex = build_table(payload, datasets, policies, cache_sizes, args.access) + "\n"
    print(f"[kg-timing] {args.access} access, capacities {cache_sizes}, "
          f"datasets {datasets} <- {args.input}", file=sys.stderr)

    if args.output:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(f"[kg-timing] wrote {out}", file=sys.stderr)
    else:
        print(tex)


if __name__ == "__main__":
    main()
