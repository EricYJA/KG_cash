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

--kg-backend picks where both stages read the KG. It now defaults to the live
Freebase on Virtuoso, the same endpoint ToG queries. Until this flag existed RoG
only ever read the per-question subgraph bundled in each HuggingFace row -- an
oracle built by someone who already knew the answer -- so a run against
`--kg-backend dataset` and a run against `virtuoso` are answering meaningfully
different questions, and the earlier artifacts tagged *_virtuoso_* are all
`dataset` runs. Two instances can run side by side on different backends:

    ./scripts/run_rog_cache_experiment.py --kg-backend virtuoso --run-tag rog_virt &
    ./scripts/run_rog_cache_experiment.py --kg-backend oxigraph --run-tag rog_oxi &

    ./scripts/run_rog_cache_experiment.py                                # 50 questions, all policies, API, live Virtuoso
    ./scripts/run_rog_cache_experiment.py --limit 200 --vendor tamu
    ./scripts/run_rog_cache_experiment.py --kg-backend dataset           # reproduce the pre-flag runs
    ./scripts/run_rog_cache_experiment.py --policies none,semantic_lru --threshold 0.85
    ./scripts/run_rog_cache_experiment.py --engine local --quant 8bit    # old fine-tuned path (dataset KG only)
    ./scripts/run_rog_cache_experiment.py --dataset RoG-cwq --limit all

Each policy starts from a COLD cache and makes a single pass over the split, so the
cache fills as it goes -- no pre-warming, no train/test leak. --keep-cache (env
KEEP_CACHE=1) keeps entries so a re-run replays against a WARM cache.

Restartable: re-using a --run-tag RESUMES by default. Completed policies (marked by
<tag>.done) are skipped, and an interrupted policy continues where it stopped --
stage 1 and stage 2 append to their JSONL, skipping questions already answered, and
the partial cache on disk is kept. Pass --fresh (env FRESH=1) to wipe the tag's
outputs/caches and redo everything from scratch.
"""
from __future__ import annotations

import argparse
import json
import os

from _rog_common import (KEYS_PATH, MODEL_NAME, MODEL_PATH, REPO_ROOT,
                         add_kg_backend_arg, assert_stages_agree,
                         docker_build, ensure_kg_backend, load_dotenv,
                         load_env_keys, make_rog_runner, add_env_file_arg, preload_dotenv,
                         quant_flag, split_for, stop_rog_server)

OUT_HOST = REPO_ROOT / "artifacts" / "rog_cache"


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
    p.add_argument("--fresh", action="store_true", default=env("FRESH", "0") == "1",
                   help="wipe this run-tag's outputs/caches and start over. Default "
                        "resumes: re-using a --run-tag skips completed policies and "
                        "continues partial stages where they stopped.")
    p.add_argument("--run-tag", default=env("RUN_TAG", ""),
                   help="subdirectory under artifacts/rog_cache/, so a second "
                        "concurrent instance does not overwrite this run")
    add_kg_backend_arg(p)
    args = p.parse_args()

    is_api = args.engine == "api"
    split = split_for(args.limit)
    qflag = [] if is_api else quant_flag(args.quant)
    policies = args.policies.replace(",", " ").split()
    # Everything below writes under /out, which is this dir bind-mounted into the
    # container -- including the per-policy caches, which are deleted on start.
    out_host = OUT_HOST / args.run_tag if args.run_tag else OUT_HOST

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys() if is_api else 0
    needs_single_key = is_api and n_keys == 0
    load_dotenv(required=("HF_TOKEN", "LLM_API_KEY") if needs_single_key else ("HF_TOKEN",))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; "
              f"the client falls back to the next one on a 400/5xx")
    docker_build()
    # Both stages talk to this, so it comes up before either runs. --network host
    # (see _rog_common.make_rog_runner) is what lets the container reach a server
    # published on the host's localhost.
    kg_endpoint = ensure_kg_backend(args.kg_backend)
    if not is_api and args.kg_backend != "dataset":
        # gen_rule_path_cached.py / predict_answer_seeded.py are the fine-tuned
        # GPU scripts and read sample["graph"] directly; they have no --kg-backend.
        p.error("--engine local only supports --kg-backend dataset")
    # Only the GPU pipeline needs the card; see run_rog_eval.py.
    if not is_api:
        stop_rog_server()
    out_host.mkdir(parents=True, exist_ok=True)

    extra_env = {"TOG_CACHE_DIR": "/togcache"}
    if is_api:
        if os.environ.get("LLM_API_KEY"):
            extra_env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]
        if n_keys:
            # Mounted rather than passed as -e values: the container reads the
            # same file the host does, so the pool has exactly one parser.
            extra_env["LLM_KEYS_FILE"] = "/keys/.env_keys"
        # cwd is /rog (read-only mount); send failed-request dumps to the writable
        # /out bind-mount so they survive the --rm container for debugging.
        extra_env["LLM_DUMP_DIR"] = "/out"

    # Passed to BOTH stages: stage 1 plans against this KG and stage 2 grounds
    # against it. Splitting them would produce a run that is neither backend.
    kg_flags = ["--kg-backend", args.kg_backend]
    if kg_endpoint:
        kg_flags += ["--kg-endpoint", kg_endpoint]

    home = os.path.expanduser("~")
    rog = make_rog_runner(
        gpus=not is_api,  # the API pipeline puts no model on the GPU
        use_user=True,
        # /togcache on the path so the API scripts import llm_client/llm_config.
        pythonpath="/rog/src:/rog/src/utils:/rogcache:/togcache",
        extra_env=extra_env,
        mounts=[
            *([f"{KEYS_PATH}:/keys/.env_keys:ro"] if n_keys else []),
            f"{REPO_ROOT}/ref_KG_projects/RoG:/rog",
            f"{REPO_ROOT}/src/RoG-cache:/rogcache",
            f"{REPO_ROOT}/src/ToG-cache/ToG:/togcache:ro",
            f"{out_host}:/out",
            f"{home}/.cache/huggingface:/hf",
        ],
    )

    for policy in policies:
        tag = policy if policy in ("none", "exact") else f"{policy}_t{args.threshold}"
        # Resume by default: a completed policy (marker present) is skipped; --fresh
        # clears the marker so the policy is redone from scratch.
        done_marker = out_host / f"{tag}.done"
        if args.fresh:
            done_marker.unlink(missing_ok=True)
        elif done_marker.exists():
            print(f"\n>>> POLICY={policy} already complete (tag={tag}); skipping. "
                  f"Pass --fresh to redo.")
            continue

        # --fresh overwrites stage 1/2 outputs; default (resume) appends, so each
        # stage skips questions already in its JSONL.
        force_flag = ["--force"] if args.fresh else []

        if policy == "none":
            cache_flags = ["--no-question-cache"]
        else:
            cache_flags = ["--cache-policy", policy,
                           "--similarity-threshold", args.threshold,
                           "--question-cache-capacity", args.capacity,
                           "--question-cache-path", f"/out/cache/{tag}.json"]
            # Each policy makes a single cold-cache pass. --fresh clears any prior
            # file so the pass starts empty; on a resume (default) the partial cache
            # on disk is exactly that cold cache filled up to the interruption, so
            # keep it and continue.
            if args.fresh and not args.keep_cache:
                (out_host / "cache" / f"{tag}.json").unlink(missing_ok=True)

        rule_dir = f"/out/gen_rule_path/{tag}/{args.dataset}/{MODEL_NAME}/{split}"
        rule_path = f"{rule_dir}/predictions_{args.n_beam}_False.jsonl"

        backend = f"api:{args.vendor}" if is_api else f"local:{args.quant}"
        print("\n" + "=" * 64)
        print(f">>> POLICY={policy}  [{split}, beam={args.n_beam}, {backend}, "
              f"kg={args.kg_backend}, threshold={args.threshold}]")
        print("=" * 64)

        print(">>> STAGE 1/3  planner (CACHED)   question -> relation paths")
        if is_api:
            model_opt = ["--model", args.model] if args.model else []
            rog(["python", "/rogcache/gen_rule_path_api.py",
                 "--model_name", MODEL_NAME, "-d", args.dataset, "--split", split,
                 "--n_beam", args.n_beam, "--vendor", args.vendor, *model_opt,
                 "--output_path", f"/out/gen_rule_path/{tag}",
                 "--timing-log", "/out/cache_timing.jsonl", *force_flag, *cache_flags,
                 *kg_flags])
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
                 "--predict_path", f"/out/KGQA/{tag}", "--seed", args.seed, *force_flag,
                 *kg_flags])
        else:
            rog(["python", "/rogcache/predict_answer_seeded.py",
                 "--model_name", MODEL_NAME, "--model_path", MODEL_PATH,
                 "-d", args.dataset, "--split", split,
                 "--prompt_path", "prompts/llama2_predict.txt",
                 "--add_rule", "--rule_path", rule_path,
                 "--predict_path", f"/out/KGQA/{tag}", "--seed", args.seed, *qflag, *force_flag])

        # predict_answer.py derives its output dir from the rule path (maps '/' and
        # '.' to '_'; basename must stay exactly predictions.jsonl).
        rule_postfix = rule_path.translate(str.maketrans({"/": "_", ".": "_"}))
        predict_file = f"/out/KGQA/{tag}/{args.dataset}/{MODEL_NAME}/{split}/{rule_postfix}/predictions.jsonl"

        # Before scoring: a run whose two stages disagree about the model is not
        # attributable to either, and its numbers should never reach summary.csv.
        print("\n>>> model check   (planner vs reasoner)")
        assert_stages_agree(
            out_host / "gen_rule_path" / tag / args.dataset / MODEL_NAME / split
            / "cache_stats.json",
            out_host / "KGQA" / tag / args.dataset / MODEL_NAME / split
            / rule_postfix / "run_config.json",
            args.model,
        )

        print("\n>>> STAGE 3/3  scoring   (Hits@1 / F1)")
        rog(["python", "src/qa_prediction/evaluate_results.py", "-d", predict_file, "--cal_f1"])

        # Point the summarizer at this config's scores. /out is bind-mounted to
        # out_host, so we can write the manifest directly on the host; the paths
        # inside stay container-relative because summarize runs in-container.
        manifest = {
            "tag": tag,
            "policy": policy,
            # In the manifest, not just the run tag: the tag is free text and the
            # existing artifacts prove it can name a backend the run never used.
            "kg_backend": args.kg_backend,
            "kg_endpoint": kg_endpoint,
            "eval_file": predict_file.replace("predictions.jsonl", "eval_result.txt"),
            "cache_stats": f"{rule_dir}/cache_stats.json",
            # Per-question times from BOTH stages. The summarizer joins them on
            # question id to get whole-question timings, which is the only way
            # this pipeline can report a full-system speedup: stage 1 alone
            # measures the planner, and the cache is a planner cache, so a
            # stage-1 speedup says nothing about the reasoner half that always
            # runs. Sidecar naming follows cache_metrics.metrics_sidecar_path.
            "stage1_metrics": f"{rule_path}.metrics.jsonl",
            "stage2_metrics": f"{predict_file}.metrics.jsonl",
            "predict_file": predict_file,
        }
        (out_host / f"manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
        done_marker.write_text("")  # policy fully scored: skip it on a same-tag resume

    print("\n" + "=" * 64)
    print(">>> SUMMARY")
    print("=" * 64)
    rog(["python", "/rogcache/summarize_rog_cache.py", "--results-dir", "/out"])

    print(f"\n>>> raw outputs under {out_host.relative_to(REPO_ROOT)}/")
    if not is_api:
        print(">>> restart the server with: docker compose up -d rog")


if __name__ == "__main__":
    main()
