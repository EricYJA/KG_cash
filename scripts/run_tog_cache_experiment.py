#!/usr/bin/env python3
"""ToG + question caching: the real accuracy experiment.

Drives compare_cache_accuracy.py, which runs four configurations end to end over
the Freebase KG and scores each with eval.py (Exact Match):
    1. main_freebase.py       --no-question-cache     baseline_main
    2. main_freebase.py       cache enabled           cache_main
    3. main_freebase_loop.py  --no-question-cache     baseline_loop
    4. main_freebase_loop.py  cache enabled           cache_loop

A cache hit skips the Virtuoso traversal plus the per-loop scoring LLM calls; the
final-answer LLM call still runs, so any Exact-Match delta comes only from reusing
the cached reasoning chain. The loop configs run the split twice so the cache
warms up (cache_loop's second pass should be near-100% hit).

Requires:
    - the KG_cash conda env (override with --conda-env / CONDA_ENV)
    - Virtuoso serving Freebase on localhost:8890 (started here via docker compose)
    - LLM_API_KEY in .env (used by the tamu/openai/google client)

Examples:
    ./scripts/run_tog_cache_experiment.py
    ./scripts/run_tog_cache_experiment.py --dataset cwq --limit 50
    ./scripts/run_tog_cache_experiment.py --vendor openai --threshold 0.85
"""
from __future__ import annotations

import argparse
import os

from _tog_common import TOG_DIR, ensure_virtuoso, load_dotenv, run_py


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conda-env", default=env("CONDA_ENV", "KG_cash"))
    p.add_argument("--dataset", default=env("DATASET", "webqsp"),
                   help="webqsp | cwq | qald | ...")
    p.add_argument("-n", "--limit", default=env("N", "20"),
                   help="samples per config (keeps token cost bounded)")
    p.add_argument("--vendor", default=env("VENDOR", "tamu"), help="tamu | openai | google")
    p.add_argument("--depth", default=env("DEPTH", "3"))
    p.add_argument("--width", default=env("WIDTH", "3"))
    p.add_argument("--threshold", default=env("THRESHOLD", "0.90"),
                   help="cosine threshold for the semantic cache")
    p.add_argument("--loop", default=env("LOOP", "2"), help="loop count for main_freebase_loop.py")
    args, extra = p.parse_known_args()

    load_dotenv(required=("LLM_API_KEY",))
    ensure_virtuoso()

    print("\n" + "=" * 64)
    print(f">>> ToG cache experiment  [dataset={args.dataset}, N={args.limit}, "
          f"vendor={args.vendor},")
    print(f">>>   depth={args.depth}, width={args.width}, threshold={args.threshold}, "
          f"loop={args.loop}]")
    print("=" * 64)

    # compare_cache_accuracy.py uses paths relative to TOG_DIR (eval.py, ../output).
    run_py(
        [
            "compare_cache_accuracy.py",
            "--dataset", args.dataset,
            "--test-limit", args.limit,
            "--vendor", args.vendor,
            "--depth", args.depth,
            "--width", args.width,
            "--similarity-threshold", args.threshold,
            "--loop", args.loop,
            *extra,
        ],
        cwd=TOG_DIR,
        conda_env=args.conda_env,
    )
    print("\n>>> per-config JSONL + summary.json under src/ToG-cache/output/compare_results/")


if __name__ == "__main__":
    main()
