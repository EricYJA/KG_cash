#!/usr/bin/env python3
"""Run ToG's own Freebase eval pipeline with the cache OFF (a clean baseline).

    stage 1  main_freebase.py --no-question-cache   question + KG -> answer jsonl
    stage 2  eval.py                                 Exact Match

Requires the KG_cash conda env, Virtuoso on :8890, and LLM_API_KEY in .env
(see run_tog_cache_experiment.py for the full cache experiment).

Examples:
    ./scripts/run_tog_eval.py
    ./scripts/run_tog_eval.py --dataset cwq --vendor openai --limit 100
"""
from __future__ import annotations

import argparse
import os

from _tog_common import EVAL_DIR, TOG_DIR, ensure_virtuoso, load_dotenv, run_py


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conda-env", default=env("CONDA_ENV", "KG_cash"))
    p.add_argument("--dataset", default=env("DATASET", "webqsp"),
                   help="webqsp | cwq | qald | ...")
    p.add_argument("-n", "--limit", default=env("N", "20"), help="number of samples")
    p.add_argument("--vendor", default=env("VENDOR", "tamu"), help="tamu | openai | google")
    p.add_argument("--depth", default=env("DEPTH", "3"))
    p.add_argument("--width", default=env("WIDTH", "3"))
    p.add_argument("--out-file", default=env("OUT_FILE", ""),
                   help="answers JSONL (default: ../output/ToG_<dataset>_baseline.jsonl)")
    args = p.parse_args()

    # main_freebase.py resolves this relative to TOG_DIR; eval.py resolves the same
    # string relative to EVAL_DIR. Both point at src/ToG-cache/output, so `../output`
    # works from either directory.
    out_file = args.out_file or f"../output/ToG_{args.dataset}_baseline.jsonl"

    load_dotenv(required=("LLM_API_KEY",))
    ensure_virtuoso()

    print(f"\n>>> STAGE 1/2  ToG traversal + reasoning (cache OFF)  "
          f"[{args.dataset}, N={args.limit}, depth={args.depth}, width={args.width}, {args.vendor}]")
    run_py(
        [
            "main_freebase.py",
            "--dataset", args.dataset,
            "--test-limit", args.limit,
            "--vendor", args.vendor,
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
