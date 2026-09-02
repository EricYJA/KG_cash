#!/usr/bin/env python3
"""Run RoG's WebQSP eval pipeline (planning -> reasoning -> scoring) in the
kgcash/rog-eval image.

Two backends, selected with --engine (default: api):
  api   : LLM-API planner + reasoner (src/RoG-cache/*_api.py). No GPU, no
          quantization, no fine-tuned weights. Uses the same TAMU/OpenAI/Google
          endpoint as the ToG experiments. NOT a paper reproduction -- the
          planner sees the KG's candidate relations (see gen_rule_path_api.py).
  local : the fine-tuned rmanluo/RoG on the GPU, quantized (--quant). This is the
          paper's method; reference WebQSP test numbers Hits@1 ~85, F1 ~70.

--kg-backend picks where stages 1 and 2 read the KG: `virtuoso` (default) or
`oxigraph` query the live Freebase endpoint; `dataset` uses the per-question
subgraph bundled in each HuggingFace row, which is what upstream RoG does and
what every run in artifacts/ before this flag used, including the ones whose tag
says virtuoso.

    ./scripts/run_rog_eval.py                       # 50 questions, API, TAMU, live Virtuoso
    ./scripts/run_rog_eval.py --limit all --vendor tamu
    ./scripts/run_rog_eval.py --kg-backend oxigraph --run-tag oxi
    ./scripts/run_rog_eval.py --kg-backend dataset  # upstream's bundled subgraphs
    ./scripts/run_rog_eval.py --engine local --quant 8bit   # fine-tuned GPU path (dataset KG only)
    ./scripts/run_rog_eval.py --dataset RoG-cwq
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


def planner_sanity_check(rule_path_host) -> None:
    """How often does a predicted path match a ground-truth path?

    A healthy planner is well above zero here; ~0% means the planner is broken,
    not the reasoner, and the answer scores below are meaningless. The rule file
    is bind-mounted from the RoG repo, so we read it directly on the host.

    Read it against the backend that produced it. Under --kg-backend dataset both
    sides of the comparison come from the same bundled subgraph, so the number is
    flattered: the planner is choosing from a relation menu drawn from the very
    graph the gold path was found in. Against a live endpoint both sides come from
    all of Freebase, and a lower rate here is the honest one.
    """
    if not rule_path_host.exists():
        print(f"    (sanity check skipped: {rule_path_host} not found)")
        return
    hit = both = empty = 0
    with rule_path_host.open() as f:
        for line in f:
            d = json.loads(line)
            preds = {tuple(pth) for pth in d["prediction"]}
            truth = {tuple(pth) for pth in d["ground_paths"]}
            if not preds:
                empty += 1
            if not truth:            # no reachable gold path in the subgraph; not scorable
                continue
            both += 1
            if preds & truth:
                hit += 1
    pct = 100 * hit / max(both, 1)
    print(f"    path hit@beam: {hit}/{both} = {pct:.1f}%   (empty predictions: {empty})")


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
    p.add_argument("--max-hop", default=env("MAX_HOP", "2"),
                   help="[api] candidate relations within this many hops of q_entity")
    p.add_argument("--quant", default=env("QUANT", "8bit"),
                   help="[local] 8bit | 4bit | fp16")
    p.add_argument("--dataset", default=env("DATASET", "RoG-webqsp"))
    p.add_argument("--run-tag", default=env("RUN_TAG", ""),
                   help="suffix for the results dir, so a second concurrent "
                        "instance does not overwrite this run")
    add_kg_backend_arg(p)
    args = p.parse_args()

    is_api = args.engine == "api"
    split = split_for(args.limit)
    # "RoG-api" keeps API outputs from colliding with the fine-tuned "RoG" ones;
    # every results path below is keyed on model_name, so the run tag goes here.
    model_name = "RoG-api" if is_api else MODEL_NAME
    if args.run_tag:
        model_name = f"{model_name}-{args.run_tag}"
    home = os.path.expanduser("~")

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys() if is_api else 0
    needs_single_key = is_api and n_keys == 0
    load_dotenv(required=("HF_TOKEN", "LLM_API_KEY") if needs_single_key else ("HF_TOKEN",))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; "
              f"the client falls back to the next one on a 400/5xx")
    docker_build()
    kg_endpoint = ensure_kg_backend(args.kg_backend)
    if not is_api and args.kg_backend != "dataset":
        # The local engine runs upstream's gen_rule_path.py / predict_answer.py,
        # which read sample["graph"] directly and have no --kg-backend.
        p.error("--engine local only supports --kg-backend dataset")
    # Only the local engine needs the card. Stopping the server on an API run
    # would evict a concurrent local run's model for no reason.
    if not is_api:
        stop_rog_server()

    if is_api:
        api_env = {}
        if os.environ.get("LLM_API_KEY"):
            api_env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]
        if n_keys:
            api_env["LLM_KEYS_FILE"] = "/keys/.env_keys"
        rog = make_rog_runner(
            gpus=False, use_user=False,
            pythonpath="/rog/src:/rog/src/utils:/kgsrc/ToG-cache/ToG",
            extra_env=api_env,
            mounts=[
                # Mounted rather than passed as -e values: the container reads
                # the same file the host does, so the pool has one parser.
                *([f"{KEYS_PATH}:/keys/.env_keys:ro"] if n_keys else []),
                f"{REPO_ROOT}/ref_KG_projects/RoG:/rog",
                f"{REPO_ROOT}/src:/kgsrc",
                f"{home}/.cache/huggingface:/root/.cache/huggingface",
            ],
        )
    else:
        rog = make_rog_runner(
            gpus=True, use_user=False,
            pythonpath="/rog/src:/rog/src/utils",
            mounts=[
                f"{REPO_ROOT}/ref_KG_projects/RoG:/rog",
                f"{home}/.cache/huggingface:/root/.cache/huggingface",
            ],
        )

    # Both stages get these: stage 1 plans against this KG and stage 2 grounds
    # against it, and a run that mixed the two would be neither backend.
    kg_flags = ["--kg-backend", args.kg_backend]
    if kg_endpoint:
        kg_flags += ["--kg-endpoint", kg_endpoint]

    model_opt = ["--model", args.model] if (is_api and args.model) else []
    qflag = [] if is_api else quant_flag(args.quant)
    rule_path = f"results/gen_rule_path/{args.dataset}/{model_name}/{split}/predictions_{args.n_beam}_False.jsonl"
    backend = f"api:{args.vendor}" if is_api else f"local:{args.quant}"

    # gen_rule_path_api.py defaults this to a path relative to the workdir, and the
    # workdir is the same bind mount in every instance -- so two concurrent runs
    # would share one cache file and race on its rewrite. Key it on model_name,
    # which already carries --run-tag.
    cache_path = f"cache/rog_question_cache_{model_name}.json"

    print(f"\n>>> STAGE 1/3  planning  (question -> relation paths)   "
          f"[{split}, beam={args.n_beam}, {backend}, kg={args.kg_backend}]")
    if is_api:
        rog(["python", "/kgsrc/RoG-cache/gen_rule_path_api.py",
             "--model_name", model_name, "-d", args.dataset, "--split", split,
             "--n_beam", args.n_beam, "--vendor", args.vendor, "--max-hop", args.max_hop,
             "--question-cache-path", cache_path, *model_opt, *kg_flags])
    else:
        rog(["python", "src/qa_prediction/gen_rule_path.py",
             "--model_name", model_name, "--model_path", MODEL_PATH,
             "-d", args.dataset, "--split", split, "--n_beam", args.n_beam, *qflag, "--force"])

    print("\n>>> planner sanity check: how often does a predicted path match a ground-truth path?")
    planner_sanity_check(REPO_ROOT / "ref_KG_projects" / "RoG" / rule_path)

    print("\n>>> STAGE 2/3  reasoning  (paths + KG -> answer)")
    if is_api:
        rog(["python", "/kgsrc/RoG-cache/predict_answer_api.py",
             "--model_name", model_name, "-d", args.dataset, "--split", split,
             "--prompt_path", "prompts/llama2_predict.txt",
             "--add_rule", "--rule_path", rule_path, "--vendor", args.vendor, *model_opt,
             "--force", *kg_flags])
    else:
        rog(["python", "src/qa_prediction/predict_answer.py",
             "--model_name", model_name, "--model_path", MODEL_PATH,
             "-d", args.dataset, "--split", split,
             "--prompt_path", "prompts/llama2_predict.txt",
             "--add_rule", "--rule_path", rule_path, *qflag, "--force"])

    # predict_answer.py derives its output dir from the rule path (see its main()):
    # it maps '/' and '.' to '_'. The basename must stay exactly predictions.jsonl.
    rule_postfix = rule_path.translate(str.maketrans({"/": "_", ".": "_"}))
    predict_file = f"results/KGQA/{args.dataset}/{model_name}/{split}/{rule_postfix}/predictions.jsonl"

    if is_api:
        # A run whose planner and reasoner ran different models is not
        # attributable to either; catch it before the scores are printed.
        print("\n>>> model check   (planner vs reasoner)")
        rog_results = REPO_ROOT / "ref_KG_projects" / "RoG"
        assert_stages_agree(
            rog_results / "results" / "gen_rule_path" / args.dataset / model_name
            / split / "cache_stats.json",
            rog_results / predict_file.replace("/predictions.jsonl", "/run_config.json"),
            args.model,
        )

    print("\n>>> STAGE 3/3  scoring   (Hits@1 / F1)")
    rog(["python", "src/qa_prediction/evaluate_results.py", "-d", predict_file, "--cal_f1"])

    print("\n>>> done. raw outputs under ref_KG_projects/RoG/results/")
    if not is_api:
        print(">>> restart the server with: docker compose up -d rog")


if __name__ == "__main__":
    main()
