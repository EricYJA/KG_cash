#!/usr/bin/env python3
"""Hit-rate simulation for the RoG planner cache. No GPU, no LLM, runs in seconds.

Use it to pick a cosine threshold before paying for the real accuracy sweep
(run_rog_cache_experiment.py).

Examples:
    ./scripts/run_rog_cache_sim.py
    ./scripts/run_rog_cache_sim.py --dataset RoG-cwq
    ./scripts/run_rog_cache_sim.py --limit 200            # first 200 questions
    ./scripts/run_rog_cache_sim.py --run-tag t085 -- -t 0.85   # save the table
    ./scripts/run_rog_cache_sim.py -- -t 0.75,0.80,0.85 -c inf

Everything after `--` (or any unrecognized flags) passes straight through to
simulate_rog_cache.py. --run-tag saves the printed table to a file so parallel
runs don't just interleave on the terminal.
"""
from __future__ import annotations

import argparse
import os

from _rog_common import (
    REPO_ROOT,
    docker_build,
    load_dotenv,
    make_rog_runner,
)


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dataset", default=env("DATASET", "RoG-webqsp"))
    p.add_argument("--split", default=env("SPLIT", "test"))
    p.add_argument("-n", "--limit", default=env("N", ""),
                   help="use only the first N questions, or 'all' (default: all)")
    p.add_argument("--run-tag", default=env("RUN_TAG", ""),
                   help="also save the result table to artifacts/rog_cache_sim/<tag>.txt "
                        "(the sim only prints to stdout otherwise, so parallel runs interleave)")
    args, extra = p.parse_known_args()

    load_dotenv(required=("HF_TOKEN",))
    docker_build(quiet=True)

    home = os.path.expanduser("~")
    # A tagged run mounts a host output dir so the table survives as a file.
    out_host = REPO_ROOT / "artifacts" / "rog_cache_sim"
    mounts = [
        f"{REPO_ROOT}/ref_KG_projects/RoG:/rog",
        f"{REPO_ROOT}/src/RoG-cache:/rogcache",
        f"{REPO_ROOT}/src/ToG-cache/ToG:/togcache:ro",
        f"{home}/.cache/huggingface:/hf",
    ]
    out_flags: list[str] = []
    if args.run_tag:
        out_host.mkdir(parents=True, exist_ok=True)
        mounts.append(f"{out_host}:/simout")
        out_flags = ["--out", f"/simout/{args.run_tag}.txt"]

    rog = make_rog_runner(
        gpus=False,          # pure numpy sim; the GPU is not needed
        use_user=True,
        pythonpath="/rog/src:/rog/src/utils:/rogcache:/togcache",
        extra_env={"TOG_CACHE_DIR": "/togcache"},
        mounts=mounts,
    )

    limit_flags = ["--limit", args.limit] if args.limit else []
    rog(["python", "/rogcache/simulate_rog_cache.py",
         "-d", args.dataset, "--split", args.split, *limit_flags, *out_flags, *extra])

    if args.run_tag:
        print(f">>> table saved to artifacts/rog_cache_sim/{args.run_tag}.txt")


if __name__ == "__main__":
    main()
