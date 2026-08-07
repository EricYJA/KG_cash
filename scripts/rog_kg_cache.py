#!/usr/bin/env python3
"""One command for the RoG KG-request cache figure: planner -> simulate -> plot.

Produces the RoG counterpart of cache_sim_webqsp_comparison.pdf -- caching
one-hop neighbourhood lookups, not questions. The three underlying steps each
need their own interpreter, PYTHONPATH and path juggling, so this wraps them:

  1. gen_rule_path_api.py   the stage-1 planner, for the relation paths RoG
                            walks. Skipped when its predictions already exist.
                            The only step that costs LLM calls (~2.6s/question).
  2. simulate_rog_kg_cache.py  replays bfs_with_rule over those paths, recording
                            every neighbourhood expansion, and runs the cache
                            simulation. Seconds.
  3. plot_cache_sim.py      the grouped-bar figure. Seconds.

Usage:
    scripts/rog_kg_cache.py -d RoG-cwq -n 400
    scripts/rog_kg_cache.py -d RoG-webqsp -n 400      # reuses the existing run
    scripts/rog_kg_cache.py -d RoG-cwq -n 400 --policies lru,lfu,fifo,random,belady,oracle
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# The runners need sentence_transformers / rank_bm25 / datasets, which only this
# env has; plain python3 can run the plot step but nothing before it.
PYTHON = "/home/stanic/anaconda3/envs/KG_cash/bin/python"
SIM_OUT = REPO / "artifacts" / "rog_kg_cache_sim"
# Stage-1 runs already on disk, so --dataset RoG-webqsp does not re-pay for the
# planner. Anything not listed here gets a fresh planner run.
EXISTING_RUNS = {
    "RoG-webqsp": REPO / "artifacts/rog_cache/rog_cache_virtuoso_new/gen_rule_path/none"
                       / "RoG-webqsp/RoG/test/predictions_3_False.jsonl",
}


def load_dotenv() -> dict:
    """LLM_API_KEY et al from .env, without clobbering what is already exported."""
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if not dotenv.exists():
        return env
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def run(cmd: list[str], env: dict, label: str) -> None:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO, env=env)
    if result.returncode != 0:
        sys.exit(f"[rog-kg-cache] {label} failed (exit {result.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", default="RoG-cwq")
    ap.add_argument("--combined", action="store_true",
                    help="both datasets into one summary and both figures, the RoG "
                         "equivalent of cache_simulator.py --combined")
    ap.add_argument("-n", "--limit", type=int, default=400,
                    help="questions to use (default 400, matching the ToG figure)")
    ap.add_argument("--n-beam", type=int, default=3)
    ap.add_argument("--vendor", default="tamu")
    ap.add_argument("--cache-sizes", default="10,50,100,500,1000")
    ap.add_argument("--policies", default="lru,lfu,oracle")
    ap.add_argument("--max-expansions", type=int, default=20000)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="use these stage-1 predictions instead of running the planner")
    ap.add_argument("--fresh", action="store_true",
                    help="re-run the planner even if its predictions already exist")
    ap.add_argument("--python", default=PYTHON)
    args = ap.parse_args()

    env = load_dotenv()
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO / "ref_KG_projects/RoG/src"),
        str(REPO / "ref_KG_projects/RoG/src/utils"),
        str(REPO / "src/RoG-cache"),
        str(REPO / "src/ToG-cache/ToG"),
    ])
    env["TOG_CACHE_DIR"] = str(REPO / "src/ToG-cache/ToG")

    split = f"test[:{args.limit}]" if args.limit else "test"
    datasets = ["RoG-webqsp", "RoG-cwq"] if args.combined else [args.dataset]

    # Step 1 -- the only expensive one, and only for datasets we lack paths for.
    prediction_paths: dict = {}
    for dataset in datasets:
        predictions = None
        if args.predictions and not args.combined:
            predictions = args.predictions
        elif not args.fresh:
            predictions = EXISTING_RUNS.get(dataset)
        if predictions is None:
            out_root = REPO / "artifacts/rog_cache" / f"{dataset.lower()}_planner"
            predictions = (out_root / "gen_rule_path/none" / dataset / "RoG" / split
                           / f"predictions_{args.n_beam}_False.jsonl")

        have = predictions.exists() and sum(1 for _ in predictions.open()) >= (args.limit or 1)
        if have and not args.fresh:
            print(f"[rog-kg-cache] {dataset}: reusing predictions at {predictions}")
        else:
            if not env.get("LLM_API_KEY"):
                sys.exit("LLM_API_KEY is not set (put it in .env or export it); "
                         "the planner step needs it")
            print(f"[rog-kg-cache] {dataset}: planner run needed "
                  f"(~{2.6 * (args.limit or 400) / 60:.0f} min for {args.limit} questions)")
            run([args.python, "src/RoG-cache/gen_rule_path_api.py",
                 "-d", dataset, "--split", split,
                 "--n_beam", str(args.n_beam), "--vendor", args.vendor,
                 "--no-question-cache",
                 "--output_path", str(predictions.parents[3])],
                env, f"stage-1 planner: {dataset} {split}")
        prediction_paths[dataset] = predictions

    # Step 2 -- replay bfs_with_rule and simulate the cache.
    if args.combined:
        summary = SIM_OUT / "rog_kg_cache_sim_summary.json"
    else:
        summary = SIM_OUT / f"rog_kg_cache_sim_{args.dataset.lower().replace('rog-', '')}.json"
    cmd = [args.python, "src/RoG-cache/simulate_rog_kg_cache.py",
           "--datasets", ",".join(datasets),
           "--limit", str(args.limit),
           "--cache-sizes", args.cache_sizes,
           "--policies", args.policies,
           "--max-expansions", str(args.max_expansions),
           "--output", str(summary)]
    for dataset in datasets:
        cmd += ["--predictions", f"{dataset}={prediction_paths[dataset]}"]
    run(cmd, env, "KG-request cache simulation")

    # Step 3 -- the figure. Works on plain python3 too, but reuse the same one.
    run([args.python, "src/ToG-cache/ToG/plot_cache_sim.py",
         "--input", str(summary), "--output-dir", str(SIM_OUT)],
        env, "figure")

    print(f"\n[rog-kg-cache] done\n  summary: {summary}")
    for dataset in datasets:
        print(f"  figure:  {SIM_OUT / f'cache_sim_{dataset.lower()}_comparison.pdf'}")


if __name__ == "__main__":
    main()
