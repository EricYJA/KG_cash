#!/usr/bin/env python3
"""Hit-rate simulation for the RoG planner cache. No GPU, no LLM, runs in seconds.

Use it to pick a cosine threshold before paying for the real accuracy sweep
(run_rog_cache_experiment.py).

Examples:
    ./scripts/run_rog_cache_sim.py
    ./scripts/run_rog_cache_sim.py --dataset RoG-cwq
    ./scripts/run_rog_cache_sim.py -- -t 0.75,0.80,0.85 -c inf

Everything after `--` (or any unrecognized flags) passes straight through to
simulate_rog_cache.py.
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
    p.add_argument("--dataset", default=env("DATASET", "RoG-webqsp"))
    p.add_argument("--split", default=env("SPLIT", "test"))
    args, extra = p.parse_known_args()

    load_dotenv(required=("HF_TOKEN",))
    docker_build(quiet=True)

    home = os.path.expanduser("~")
    rog = make_rog_runner(
        gpus=False,          # pure numpy sim; the GPU is not needed
        use_user=True,
        pythonpath="/rog/src:/rog/src/utils:/rogcache:/togcache",
        extra_env={"TOG_CACHE_DIR": "/togcache"},
        mounts=[
            f"{REPO_ROOT}/ref_KG_projects/RoG:/rog",
            f"{REPO_ROOT}/src/RoG-cache:/rogcache",
            f"{REPO_ROOT}/src/ToG-cache/ToG:/togcache:ro",
            f"{home}/.cache/huggingface:/hf",
        ],
    )

    rog(["python", "/rogcache/simulate_rog_cache.py",
         "-d", args.dataset, "--split", args.split, *extra])


if __name__ == "__main__":
    main()
