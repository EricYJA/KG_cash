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

Two instances can run side by side on different SPARQL backends; --kg-backend
also tags the default cache/results dirs so neither run clobbers the other:
    ./scripts/run_tog_cache_experiment.py --kg-backend virtuoso &
    ./scripts/run_tog_cache_experiment.py --kg-backend oxigraph &
"""
from __future__ import annotations

import argparse
import os

from _tog_common import REPO_ROOT, TOG_DIR, add_run_args, load_dotenv, resolve_run, run_py

OUTPUT_DIR = REPO_ROOT / "src" / "ToG-cache" / "output"


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
    add_run_args(p)
    args, extra = p.parse_known_args()

    load_dotenv(required=("LLM_API_KEY",))
    endpoint, tag = resolve_run(args)

    # compare_cache_accuracy.py wipes its --cache-dir on startup and rewrites its
    # --results-dir, so two instances sharing the defaults would destroy each
    # other's runs mid-flight. Tag both, unless the caller passed their own.
    dir_flags: list[str] = []
    results_dir = "(see --results-dir)"
    if not any(a.startswith("--cache-dir") for a in extra):
        dir_flags += ["--cache-dir", str(OUTPUT_DIR / "compare_caches" / tag)]
    if not any(a.startswith("--results-dir") for a in extra):
        results_dir = str((OUTPUT_DIR / "compare_results" / tag).relative_to(REPO_ROOT))
        dir_flags += ["--results-dir", str(OUTPUT_DIR / "compare_results" / tag)]

    print("\n" + "=" * 64)
    print(f">>> ToG cache experiment  [dataset={args.dataset}, N={args.limit}, "
          f"vendor={args.vendor},")
    print(f">>>   depth={args.depth}, width={args.width}, threshold={args.threshold}, "
          f"loop={args.loop}, kg={args.kg_backend} @ {endpoint}]")
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
            *dir_flags,
            *extra,
        ],
        cwd=TOG_DIR,
        conda_env=args.conda_env,
    )
    print(f"\n>>> per-config JSONL + summary.json under {results_dir}/")


if __name__ == "__main__":
    main()
