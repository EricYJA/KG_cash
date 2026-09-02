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
the semantic cache persists -- and is rebuilt from the answers file if the cache
itself did not survive, so a resumed run measures the cache the answers were
produced with). Re-using a tag after changing the dataset, model, threshold,
capacity or --loop is refused rather than merged: those runs are not comparable
and appending them to one file would score the mixture. Pass --fresh (env
FRESH=1) to wipe and start over.
"""
from __future__ import annotations

import argparse
import os

from _tog_common import (KEYS_PATH, REPO_ROOT, TOG_DIR, add_run_args,
                         load_dotenv, load_env_keys, add_env_file_arg, preload_dotenv,
                         resolve_run, run_py)

OUTPUT_DIR = REPO_ROOT / "src" / "ToG-cache" / "output"


def main() -> None:
    # .env carries the documented defaults for --model, --dataset, --run-tag and
    # the rest (see .env.example). argparse evaluates each default when the
    # argument is declared, so .env has to be in the environment before the
    # parser below is built -- otherwise those values are ignored unless the
    # caller exported them, and `source .env` does not export.
    preload_dotenv()
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_file_arg(p)
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

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys()
    load_dotenv(required=() if n_keys else ("LLM_API_KEY",))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; "
              f"the client falls back to the next one on a 400/5xx")
    endpoint, tag = resolve_run(args)

    # Two instances sharing the default --cache-dir / --results-dir would append
    # to each other's per-policy files and score the mixture. Tag both, unless
    # the caller passed their own. (compare_cache_accuracy.py also refuses to
    # resume a results dir a different config wrote, so a reused tag now stops
    # the run instead of merging two experiments -- but only one run at a time
    # can own a tag either way.)
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
