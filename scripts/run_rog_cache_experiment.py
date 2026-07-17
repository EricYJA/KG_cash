#!/usr/bin/env python3
"""RoG + question caching, the same experiment we run for ToG.

For each cache policy, run the full RoG pipeline and score it:
    stage 1  planner   question -> relation paths   <-- THE CACHED STAGE
    stage 2  reasoner  paths + KG -> answer         (always runs)
    stage 3  scoring   Hits@1 / F1

A hit skips the planner's LLM call, exactly as a ToG chain-cache hit skips
Virtuoso plus the per-loop scoring calls. Stage 2 always runs, so any Hits@1/F1
delta is caused by the reused relation paths and nothing else.

By default this uses the LLM-API pipeline (--engine api: gen_rule_path_api.py +
predict_answer_api.py, no GPU). --engine local runs the fine-tuned RoG on the
GPU with the same cache.

    ./scripts/run_rog_cache_experiment.py                                # 50 questions, all policies, API
    ./scripts/run_rog_cache_experiment.py --limit 200 --vendor tamu
    ./scripts/run_rog_cache_experiment.py --policies none,semantic_lru --threshold 0.85
    ./scripts/run_rog_cache_experiment.py --engine local --quant 8bit    # old fine-tuned path
    ./scripts/run_rog_cache_experiment.py --dataset RoG-cwq --limit all

Each policy starts from a COLD cache and makes a single pass over the split, so the
cache fills as it goes -- no pre-warming, no train/test leak. --keep-cache (env
KEEP_CACHE=1) keeps entries so a re-run replays against a WARM cache.
"""
from __future__ import annotations

import argparse
import json
import os

from _rog_common import (
    MODEL_NAME,
    MODEL_PATH,
    REPO_ROOT,
    docker_build,
    load_dotenv,
    make_rog_runner,
    quant_flag,
    split_for,
    stop_rog_server,
)

OUT_HOST = REPO_ROOT / "artifacts" / "rog_cache"


def main() -> None:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--limit", default=env("N", "50"),
                   help="number of test questions, or 'all'")
    p.add_argument("--n-beam", default=env("N_BEAM", "3"))
    p.add_argument("--engine", default=env("ENGINE", "api"), choices=["api", "local"],
                   help="api: LLM-API planner+reasoner (default, no GPU). "
                        "local: fine-tuned RoG on the GPU.")
    p.add_argument("--vendor", default=env("VENDOR", "tamu"),
                   help="[api] tamu | openai | google")
    p.add_argument("--model", default=env("MODEL", ""),
                   help="[api] override the vendor's default model id")
    p.add_argument("--quant", default=env("QUANT", "8bit"),
                   help="[local] 8bit | 4bit | fp16")
    p.add_argument("--dataset", default=env("DATASET", "RoG-webqsp"))
    p.add_argument("--threshold", default=env("THRESHOLD", "0.90"),
                   help="cosine threshold for the semantic policies")
    p.add_argument("--capacity", default=env("CAPACITY", "4096"))
    p.add_argument("--seed", default=env("SEED", "42"),
                   help="makes stage 2 deterministic; see predict_answer_seeded.py")
    p.add_argument("--policies",
                   default=env("POLICIES", "none exact semantic_lru semantic_lfu semantic_oracle"),
                   help="space- or comma-separated policy list")
    p.add_argument("--keep-cache", action="store_true", default=env("KEEP_CACHE", "0") == "1",
                   help="keep each policy's cache so a re-run replays against a warm cache")
    args = p.parse_args()

    is_api = args.engine == "api"
    split = split_for(args.limit)
    qflag = [] if is_api else quant_flag(args.quant)
    policies = args.policies.replace(",", " ").split()

    load_dotenv(required=("HF_TOKEN", "LLM_API_KEY") if is_api else ("HF_TOKEN",))
    docker_build()
    stop_rog_server()
    OUT_HOST.mkdir(parents=True, exist_ok=True)

    extra_env = {"TOG_CACHE_DIR": "/togcache"}
    if is_api:
        extra_env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]

    home = os.path.expanduser("~")
    rog = make_rog_runner(
        gpus=not is_api,  # the API pipeline puts no model on the GPU
        use_user=True,
        # /togcache on the path so the API scripts import llm_client/llm_config.
        pythonpath="/rog/src:/rog/src/utils:/rogcache:/togcache",
        extra_env=extra_env,
        mounts=[
            f"{REPO_ROOT}/ref_KG_projects/RoG:/rog",
            f"{REPO_ROOT}/src/RoG-cache:/rogcache",
            f"{REPO_ROOT}/src/ToG-cache/ToG:/togcache:ro",
            f"{OUT_HOST}:/out",
            f"{home}/.cache/huggingface:/hf",
        ],
    )

    for policy in policies:
        tag = policy if policy in ("none", "exact") else f"{policy}_t{args.threshold}"

        if policy == "none":
            cache_flags = ["--no-question-cache"]
        else:
            cache_flags = ["--cache-policy", policy,
                           "--similarity-threshold", args.threshold,
                           "--question-cache-capacity", args.capacity,
                           "--question-cache-path", f"/out/cache/{tag}.json"]
            # Cold cache per policy unless --keep-cache: never reuse another run's entries.
            if not args.keep_cache:
                (OUT_HOST / "cache" / f"{tag}.json").unlink(missing_ok=True)

        rule_dir = f"/out/gen_rule_path/{tag}/{args.dataset}/{MODEL_NAME}/{split}"
        rule_path = f"{rule_dir}/predictions_{args.n_beam}_False.jsonl"

        backend = f"api:{args.vendor}" if is_api else f"local:{args.quant}"
        print("\n" + "=" * 64)
        print(f">>> POLICY={policy}  [{split}, beam={args.n_beam}, {backend}, "
              f"threshold={args.threshold}]")
        print("=" * 64)

        print(">>> STAGE 1/3  planner (CACHED)   question -> relation paths")
        if is_api:
            model_opt = ["--model", args.model] if args.model else []
            rog(["python", "/rogcache/gen_rule_path_api.py",
                 "--model_name", MODEL_NAME, "-d", args.dataset, "--split", split,
                 "--n_beam", args.n_beam, "--vendor", args.vendor, *model_opt,
                 "--output_path", f"/out/gen_rule_path/{tag}",
                 "--timing-log", "/out/cache_timing.jsonl", *cache_flags])
        else:
            rog(["python", "/rogcache/gen_rule_path_cached.py",
                 "--model_name", MODEL_NAME, "--model_path", MODEL_PATH,
                 "-d", args.dataset, "--split", split, "--n_beam", args.n_beam, *qflag,
                 "--output_path", f"/out/gen_rule_path/{tag}", *cache_flags])

        print("\n>>> STAGE 2/3  reasoner (uncached, seeded)   paths + KG -> answer")
        # Seeded: RoG shuffles + truncates overflowing path lists with an unseeded
        # RNG, which alone swings Hits@1 by ~10 points run-to-run and would swamp
        # the caching effect. See src/RoG-cache/predict_answer_seeded.py.
        if is_api:
            model_opt = ["--model", args.model] if args.model else []
            rog(["python", "/rogcache/predict_answer_api.py",
                 "--model_name", MODEL_NAME, "-d", args.dataset, "--split", split,
                 "--prompt_path", "prompts/llama2_predict.txt",
                 "--add_rule", "--rule_path", rule_path, "--vendor", args.vendor, *model_opt,
                 "--predict_path", f"/out/KGQA/{tag}", "--seed", args.seed, "--force"])
        else:
            rog(["python", "/rogcache/predict_answer_seeded.py",
                 "--model_name", MODEL_NAME, "--model_path", MODEL_PATH,
                 "-d", args.dataset, "--split", split,
                 "--prompt_path", "prompts/llama2_predict.txt",
                 "--add_rule", "--rule_path", rule_path,
                 "--predict_path", f"/out/KGQA/{tag}", "--seed", args.seed, *qflag, "--force"])

        # predict_answer.py derives its output dir from the rule path (maps '/' and
        # '.' to '_'; basename must stay exactly predictions.jsonl).
        rule_postfix = rule_path.translate(str.maketrans({"/": "_", ".": "_"}))
        predict_file = f"/out/KGQA/{tag}/{args.dataset}/{MODEL_NAME}/{split}/{rule_postfix}/predictions.jsonl"

        print("\n>>> STAGE 3/3  scoring   (Hits@1 / F1)")
        rog(["python", "src/qa_prediction/evaluate_results.py", "-d", predict_file, "--cal_f1"])

        # Point the summarizer at this config's scores. /out is bind-mounted to
        # OUT_HOST, so we can write the manifest directly on the host; the paths
        # inside stay container-relative because summarize runs in-container.
        manifest = {
            "tag": tag,
            "policy": policy,
            "eval_file": predict_file.replace("predictions.jsonl", "eval_result.txt"),
            "cache_stats": f"{rule_dir}/cache_stats.json",
        }
        (OUT_HOST / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 64)
    print(">>> SUMMARY")
    print("=" * 64)
    rog(["python", "/rogcache/summarize_rog_cache.py", "--results-dir", "/out"])

    print("\n>>> raw outputs under artifacts/rog_cache/")
    if not is_api:
        print(">>> restart the server with: docker compose up -d rog")


if __name__ == "__main__":
    main()
