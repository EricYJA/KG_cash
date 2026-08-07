#!/usr/bin/env python3
"""Hit-rate simulation for the ToG question/chain cache.

No LLM, no SPARQL, no Virtuoso, no GPU -- runs in seconds. Use it to pick a cosine
threshold (and see how capacity affects hit rate) before paying for the real
accuracy experiment (run_tog_cache_experiment.py).

It walks the dataset's questions in order; on a miss it inserts a dummy chain,
then reports per-(policy, capacity) hit / miss / hit_rate. That is exactly how
many ToG runs -- each a Virtuoso traversal plus the per-loop scoring LLM calls --
you would have skipped on this dataset. Twin of run_tog_cache_sim's RoG version.

Examples:
    ./scripts/run_tog_cache_sim.py
    ./scripts/run_tog_cache_sim.py --dataset cwq --limit 500
    ./scripts/run_tog_cache_sim.py --threshold 0.85 --capacities 32,128,512,inf
    ./scripts/run_tog_cache_sim.py --passes 2          # 2nd pass reveals exact-hit potential

Env-var defaults (CONDA_ENV, DATASET, N, POLICIES, CAPACITIES, THRESHOLD, PASSES)
are honored for parity with the old shell scripts. Unrecognized args pass through
to simulate_cache.py.
"""
from __future__ import annotations

import argparse
import os

from _tog_common import TOG_DIR, run_py


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conda-env", default=env("CONDA_ENV", "KG_cash"))
    p.add_argument("-d", "--dataset", default=env("DATASET", "webqsp"),
                   help="webqsp | cwq | qald | lcquad | lcquad_test | ...")
    p.add_argument("-n", "--limit", default=env("N", "") or None,
                   help="number of questions (default: all)")
    p.add_argument("-p", "--policies",
                   default=env("POLICIES", "exact,semantic_lru,semantic_lfu,semantic_oracle"))
    p.add_argument("-c", "--capacities", default=env("CAPACITIES", "32,128,512,2048,inf"))
    p.add_argument("-t", "--threshold", default=env("THRESHOLD", "0.90"))
    p.add_argument("--passes", default=env("PASSES", "1"))
    args, extra = p.parse_known_args()

    sim_args = [
        "simulate_cache.py",
        "-d", args.dataset,
        "-p", args.policies,
        "-c", args.capacities,
        "-t", args.threshold,
        "--passes", args.passes,
    ]
    if args.limit:
        sim_args += ["-n", args.limit]
    sim_args += extra

    print(f">>> ToG cache hit-rate simulation  "
          f"[dataset={args.dataset}, threshold={args.threshold}, passes={args.passes}]")
    # simulate_cache.py os.chdir()s to its own dir to load the dataset; run from
    # TOG_DIR anyway to match the other runners.
    run_py(sim_args, cwd=TOG_DIR, conda_env=args.conda_env)


if __name__ == "__main__":
    main()
