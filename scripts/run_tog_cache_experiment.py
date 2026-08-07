#!/usr/bin/env python3
"""ToG + question caching: the real accuracy experiment.

Drives compare_cache_accuracy.py, which sweeps the cache policies (default run
order: semantic_lru semantic_lfu none exact semantic_oracle) end to end over the
Freebase KG and scores each with eval.py (Exact Match / Hits@1 / F1). This covers
the same policies as the RoG experiment, so scripts/summarize_tog_cache.py +
plot_rog_cache_results.py can put ToG and RoG on the same per-policy figure.

'none' is the uncached baseline; every other policy runs the same single pass with
that cache enabled (cold). A cache hit skips the Virtuoso traversal plus the
per-loop scoring LLM calls; the final-answer LLM call still runs, so any accuracy
delta comes only from reusing the cached reasoning chain. Pass --loop N (N>1) to
warm the cache across N passes per policy instead of a single pass.

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

Restartable: re-using a --run-tag RESUMES by default. Completed configs (marked by
<config>.done) are skipped, and an interrupted config continues where it stopped
(main_freebase.py / main_freebase_loop.py skip questions already in their JSONL, and
the semantic cache persists). Pass --fresh (env FRESH=1) to wipe and start over.
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
    p.add_argument("--model", default=env("MODEL", ""),
                   help="override the vendor's default model id")
    p.add_argument("--depth", default=env("DEPTH", "3"))
    p.add_argument("--width", default=env("WIDTH", "3"))
    p.add_argument("--threshold", default=env("THRESHOLD", "0.90"),
                   help="cosine threshold for the semantic cache")
    p.add_argument("--capacity", default=env("CAPACITY", "4096"),
                   help="max cached questions per policy (LRU/LFU eviction); "
                        "the uncached 'none' baseline ignores it")
    p.add_argument("--policies",
                   default=env("POLICIES", "semantic_lru semantic_lfu none exact semantic_oracle"),
                   help="cache policies to sweep, in run order (space- or "
                        "comma-separated)")
    p.add_argument("--loop", default=env("LOOP", "1"),
                   help="passes per policy (1 = single pass, matching RoG; "
                        ">1 warms the cache via main_freebase_loop.py)")
    p.add_argument("--fresh", action="store_true", default=env("FRESH", "0") == "1",
                   help="wipe this run-tag's caches/outputs and start over. Default "
                        "resumes: re-using a run-tag continues where it stopped.")
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
          f"vendor={args.vendor}, model={args.model or '(vendor default)'},")
    print(f">>>   depth={args.depth}, width={args.width}, threshold={args.threshold}, "
          f"capacity={args.capacity}, loop={args.loop}, kg={args.kg_backend} @ {endpoint}]")
    print(f">>>   policies: {args.policies}")
    print("=" * 64)

    # compare_cache_accuracy.py uses paths relative to TOG_DIR (eval.py, ../output).
    run_py(
        [
            "compare_cache_accuracy.py",
            "--dataset", args.dataset,
            "--test-limit", args.limit,
            "--vendor", args.vendor,
            *(["--model", args.model] if args.model else []),
            "--depth", args.depth,
            "--width", args.width,
            "--similarity-threshold", args.threshold,
            "--capacity", args.capacity,
            "--policies", args.policies,
            "--loop", args.loop,
            *(["--fresh"] if args.fresh else []),
            *dir_flags,
            *extra,
        ],
        cwd=TOG_DIR,
        conda_env=args.conda_env,
    )
    print(f"\n>>> per-config JSONL + summary.json under {results_dir}/")


if __name__ == "__main__":
    main()
