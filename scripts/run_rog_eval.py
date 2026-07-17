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

    ./scripts/run_rog_eval.py                       # 50 questions, API, TAMU
    ./scripts/run_rog_eval.py --limit all --vendor tamu
    ./scripts/run_rog_eval.py --engine local --quant 8bit   # fine-tuned GPU path
    ./scripts/run_rog_eval.py --dataset RoG-cwq
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


def planner_sanity_check(rule_path_host) -> None:
    """How often does a predicted path match a ground-truth path?

    A healthy planner is well above zero here; ~0% means the planner is broken,
    not the reasoner, and the answer scores below are meaningless. The rule file
    is bind-mounted from the RoG repo, so we read it directly on the host.
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
    p.add_argument("--max-hop", default=env("MAX_HOP", "2"),
                   help="[api] candidate relations within this many hops of q_entity")
    p.add_argument("--quant", default=env("QUANT", "8bit"),
                   help="[local] 8bit | 4bit | fp16")
    p.add_argument("--dataset", default=env("DATASET", "RoG-webqsp"))
    args = p.parse_args()

    is_api = args.engine == "api"
    split = split_for(args.limit)
    # "RoG-api" keeps API outputs from colliding with the fine-tuned "RoG" ones.
    model_name = "RoG-api" if is_api else MODEL_NAME
    home = os.path.expanduser("~")

    load_dotenv(required=("HF_TOKEN", "LLM_API_KEY") if is_api else ("HF_TOKEN",))
    docker_build()
    stop_rog_server()

    if is_api:
        rog = make_rog_runner(
            gpus=False, use_user=False,
            pythonpath="/rog/src:/rog/src/utils:/kgsrc/ToG-cache/ToG",
            extra_env={"LLM_API_KEY": os.environ["LLM_API_KEY"]},
            mounts=[
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

    model_opt = ["--model", args.model] if (is_api and args.model) else []
    qflag = [] if is_api else quant_flag(args.quant)
    rule_path = f"results/gen_rule_path/{args.dataset}/{model_name}/{split}/predictions_{args.n_beam}_False.jsonl"
    backend = f"api:{args.vendor}" if is_api else f"local:{args.quant}"

    print(f"\n>>> STAGE 1/3  planning  (question -> relation paths)   "
          f"[{split}, beam={args.n_beam}, {backend}]")
    if is_api:
        rog(["python", "/kgsrc/RoG-cache/gen_rule_path_api.py",
             "--model_name", model_name, "-d", args.dataset, "--split", split,
             "--n_beam", args.n_beam, "--vendor", args.vendor, "--max-hop", args.max_hop, *model_opt])
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
             "--add_rule", "--rule_path", rule_path, "--vendor", args.vendor, *model_opt, "--force"])
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

    print("\n>>> STAGE 3/3  scoring   (Hits@1 / F1)")
    rog(["python", "src/qa_prediction/evaluate_results.py", "-d", predict_file, "--cal_f1"])

    print("\n>>> done. raw outputs under ref_KG_projects/RoG/results/")
    if not is_api:
        print(">>> restart the server with: docker compose up -d rog")


if __name__ == "__main__":
    main()
