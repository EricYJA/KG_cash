#!/usr/bin/env python3
"""Run ToG's own Freebase eval pipeline with the cache OFF (a clean baseline).

    stage 1  main_freebase.py --no-question-cache   question + KG -> answer jsonl
    stage 2  eval.py                                 Exact Match

Requires the KG_cash conda env, Virtuoso on :8890, and LLM_API_KEY in .env
(see run_tog_cache_experiment.py for the full cache experiment).

Examples:
    ./scripts/run_tog_eval.py
    ./scripts/run_tog_eval.py --dataset cwq --vendor openai --limit 100

Two instances can run side by side on different SPARQL backends; --kg-backend
also tags the default output file so neither run clobbers the other:
    ./scripts/run_tog_eval.py --kg-backend virtuoso &
    ./scripts/run_tog_eval.py --kg-backend oxigraph &
"""
from __future__ import annotations

import argparse
import os

from _tog_common import (EVAL_DIR, KEYS_PATH, TOG_DIR, add_run_args,
                         load_dotenv, load_env_keys, add_env_file_arg, preload_dotenv,
                         resolve_run, run_py)


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
    p.add_argument("-n", "--limit", default=env("N", "20"), help="number of samples")
    p.add_argument("--vendor", default=env("VENDOR", "tamu"), help="tamu | openai | google")
    p.add_argument("--model", default=env("MODEL", ""),
                   help="override the vendor's default model id. Without it every "
                        "stage runs the vendor preset (tamu -> Claude-Haiku), which "
                        "is what a run labelled for another model would silently be.")
    p.add_argument("--depth", default=env("DEPTH", "3"))
    p.add_argument("--width", default=env("WIDTH", "3"))
    p.add_argument("--out-file", default=env("OUT_FILE", ""),
                   help="answers JSONL (default: ../output/ToG_<dataset>_baseline_<tag>.jsonl)")
    add_run_args(p)
    args = p.parse_args()

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys()
    load_dotenv(required=() if n_keys else ("LLM_API_KEY",))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; "
              f"the client falls back to the next one on a 400/5xx")
    endpoint, tag = resolve_run(args)

    # main_freebase.py resolves this relative to TOG_DIR; eval.py resolves the same
    # string relative to EVAL_DIR. Both point at src/ToG-cache/output, so `../output`
    # works from either directory. The tag keeps a concurrent instance on another
    # backend from overwriting this run's answers.
    out_file = args.out_file or f"../output/ToG_{args.dataset}_baseline_{tag}.jsonl"

    print(f"\n>>> STAGE 1/2  ToG traversal + reasoning (cache OFF)  "
          f"[{args.dataset}, N={args.limit}, depth={args.depth}, width={args.width}, "
          f"{args.vendor}/{args.model or '(vendor default)'}, "
          f"kg={args.kg_backend} @ {endpoint}]")
    run_py(
        [
            "main_freebase.py",
            "--dataset", args.dataset,
            "--test-limit", args.limit,
            "--vendor", args.vendor,
            # ToG makes one model do every stage -- relation pruning, entity
            # pruning, the sufficiency check and the final answer all go through
            # utils.run_llm -- so this single flag covers planning and answering.
            *(["--model", args.model] if args.model else []),
            "--depth", args.depth,
            "--width", args.width,
            "--no-question-cache",
            "--output-file", out_file,
        ],
        cwd=TOG_DIR,
        conda_env=args.conda_env,
    )

    print("\n>>> STAGE 2/2  scoring (Exact Match)")
    # eval.py reads ground truth from ../data and must run from EVAL_DIR.
    run_py(
        ["eval.py", "--dataset", args.dataset, "--output_file", out_file],
        cwd=EVAL_DIR,
        conda_env=args.conda_env,
    )
    print(f"\n>>> baseline answers under src/ToG-cache/output/  ({os.path.basename(out_file)})")


if __name__ == "__main__":
    main()
