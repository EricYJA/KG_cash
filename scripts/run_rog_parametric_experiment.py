#!/usr/bin/env python3
"""RoG with the knowledge graph removed: the parametric-knowledge ablation.

Same workflow as scripts/run_rog_cache_experiment.py, same dataset, same split,
same vendor, same scorer -- with the knowledge graph taken out:

    stage 1  planner   question -> relation paths      <-- REMOVED (and with it
                                                           the cache under test)
    stage 2  reasoner  question -> answer              (no paths, no triples)
    stage 3  scoring   Hits@1 / F1                     (identical to the KG run)

Nothing from the KG reaches the model: no relation paths, no grounded triples,
no subgraph, no topic entities. The reasoner prompt is changed to license the
model's own parametric knowledge instead of forbidding it (see
src/RoG-cache/predict_answer_parametric.py), because with no paths in the prompt
the grounded run's "answer only from the paths" rule would forbid answering at
all. Grounding stays off in every other sense of the word: the client posts a
plain chat-completions payload with no tools, no retrieval and no vendor
search-grounding, so the weights are the only knowledge source in play.

The point is the ABLATION printed at the end: this run's score next to the cache
experiment's KG-grounded rows, and the delta between them. That delta is what
the knowledge graph is actually worth on this split -- the part of the grounded
score the model could not have produced from memory. Cache policies are worth
comparing only above this floor.

    ./scripts/run_rog_parametric_experiment.py                        # 50 questions
    ./scripts/run_rog_parametric_experiment.py --limit all --vendor tamu
    ./scripts/run_rog_parametric_experiment.py --dataset RoG-cwq --limit 200
    ./scripts/run_rog_parametric_experiment.py --compare-dir artifacts/rog_cache/my_run

Outputs land under artifacts/rog_parametric/ (add --run-tag for a subdirectory),
never under artifacts/rog_cache/, so a cache run and its ablation cannot
overwrite each other.

Restartable on the same terms as the cache experiment: re-using a --run-tag
RESUMES (a finished run is marked by parametric.done and is skipped; an
interrupted one continues, since stage 2 appends to its JSONL and skips
questions already answered). --fresh (env FRESH=1) wipes the tag and redoes it.

API engine only: the ablation asks what a chat model knows unaided, and the
local fine-tuned RoG checkpoint is trained to read answers off supplied paths,
so running it with none would measure the fine-tune's confusion, not knowledge.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from _ablation import (emit_ablation, guard_run_identity,  # noqa: E402
                       resolve_model, warn_tag_model_mismatch)
from _rog_common import (  # noqa: E402
    KEYS_PATH, MODEL_NAME, REPO_ROOT, docker_build, load_dotenv, load_env_keys,
    make_rog_runner, add_env_file_arg, preload_dotenv, split_for,
)

OUT_HOST = REPO_ROOT / "artifacts" / "rog_parametric"
# Where run_rog_cache_experiment.py leaves the KG-grounded rows to ablate against.
CACHE_OUT_HOST = REPO_ROOT / "artifacts" / "rog_cache"

PARAMETRIC = "parametric_no_kg"

# RoG's evaluate_results.py writes this one line to eval_result.txt, on a 0-100
# scale. Same regex as summarize_rog_cache.py; duplicated rather than imported
# because that module lives in the container image's tree and this runs on the
# host, where the ML imports around it are not available.
METRIC_RE = re.compile(
    r"Accuracy:\s*([\d.]+)\s+Hit:\s*([\d.]+)\s+F1:\s*([\d.]+)\s+"
    r"Precision:\s*([\d.]+)\s+Recall:\s*([\d.]+)"
)

# (row key, header, format). RoG scores 0-100; n and seconds are counts.
ABLATION_COLUMNS = [
    ("n_questions", "n", ">8"),
    ("hit", "Hits@1", ">10.1f"),
    ("f1", "F1", ">8.1f"),
    ("precision", "Prec", ">8.1f"),
    ("recall", "Recall", ">8.1f"),
    ("accuracy", "Acc", ">8.1f"),
    ("wall_s_total", "wall_s", ">10.1f"),
]
# Deliberately not a column: LLM calls. The cache run counts them per stage
# (planner_llm_calls), this run has no stages to split, and putting the two in
# one column would compare a planner's calls against a whole question's.


def parse_eval_file(path: Path) -> dict:
    """Pull the five metrics out of RoG's eval_result.txt, or {} if unreadable."""
    if not path.exists():
        return {}
    match = METRIC_RE.search(path.read_text())
    if not match:
        return {}
    accuracy, hit, f1, precision, recall = (float(g) for g in match.groups())
    return {"accuracy": accuracy, "hit": hit, "f1": f1,
            "precision": precision, "recall": recall}


def read_sidecar(path: Path) -> dict:
    """Sum the per-question timing sidecar written by install_stage2_timing().

    Stdlib only (the container's rog_e2e_metrics is not importable here), and
    tolerant of a truncated last line: the sidecar is appended to as the run
    goes, so a resumed run's file can end mid-record.

    De-duplicated by question id, last record wins -- the same rule
    cache_metrics.read_question_metrics() and rog_e2e_metrics.load_sidecar()
    apply, and the reason this is restart-safe. A question whose reasoner call
    failed is left out of predictions.jsonl, so the next resume redoes it and
    appends a SECOND record for it; summing both would inflate n_questions past
    the split and charge its time twice against the ablation. A record with no
    id keeps its position as its identity rather than collapsing the others.
    """
    if not path.exists():
        return {}
    records: dict = {}
    for index, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        records[rec.get("id") or f"#{index}"] = rec
    return {
        "n_questions": len(records),
        "llm_calls": sum(r.get("llm_calls", 0) for r in records.values()),
        "wall_s_total": round(
            sum(r.get("elapsed_s", 0.0) for r in records.values()), 3
        ),
    }


def resolve_container_path(path_str: str, results_dir: Path) -> Path:
    """Re-root a container-absolute path (/out/...) onto its host results dir.

    The cache experiment's manifests are written from inside the container, so
    their paths start at /out. Here we read them from the host, where /out does
    not exist; the tail of the path is the same either way.
    """
    path = Path(path_str)
    if path.exists():
        return path
    parts = path.parts
    for i in range(len(parts)):
        candidate = results_dir.joinpath(*parts[i:])
        if candidate.exists():
            return candidate
    return path


def load_kg_rows(compare_dir: Path) -> list[dict]:
    """One row per KG-grounded policy from a finished cache-experiment run.

    Prefers summary.csv, which run_rog_cache_experiment.py's summarizer writes at
    the end of a complete sweep. Falls back to the per-policy manifests, so a
    sweep that was interrupted before the summary step still contributes the
    policies it did finish rather than nothing at all.
    """
    rows: list[dict] = []
    summary_csv = compare_dir / "summary.csv"
    if summary_csv.exists():
        with summary_csv.open(newline="") as f:
            for raw in csv.DictReader(f):
                row = {"condition": raw.get("policy") or raw.get("tag"),
                       "tag": raw.get("tag")}
                for key, _, _ in ABLATION_COLUMNS:
                    value = raw.get(key, "")
                    if value not in ("", None):
                        try:
                            row[key] = float(value)
                        except ValueError:
                            continue
                rows.append(row)
        if rows:
            return rows

    for manifest_path in sorted(compare_dir.glob("manifest_*.json")):
        manifest = json.loads(manifest_path.read_text())
        row = {"condition": manifest.get("policy"), "tag": manifest.get("tag")}
        row.update(parse_eval_file(
            resolve_container_path(manifest["eval_file"], compare_dir)))
        predict_file = manifest.get("predict_file")
        if predict_file:
            sidecar = resolve_container_path(f"{predict_file}.metrics.jsonl", compare_dir)
            # Question count only. This sidecar times stage 2 alone, and the
            # parametric row's wall time is a whole question, so filling
            # wall_s_total from it would put half a question next to a whole one.
            # summary.csv's joined end-to-end time (the branch above) is the one
            # that is comparable.
            row.update({k: v for k, v in read_sidecar(sidecar).items()
                        if k == "n_questions"})
        rows.append(row)
    return rows


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
    p.add_argument("--vendor", default=env("VENDOR", "tamu"), help="tamu | openai | google")
    p.add_argument("--model", default=env("MODEL", ""),
                   help="override the vendor's default model id")
    p.add_argument("--dataset", default=env("DATASET", "RoG-webqsp"))
    p.add_argument("--seed", default=env("SEED", "42"),
                   help="pinned to match the KG-grounded run it is compared to")
    p.add_argument("--fresh", action="store_true", default=env("FRESH", "0") == "1",
                   help="wipe this run-tag's outputs and start over. Default resumes.")
    p.add_argument("--run-tag", default=env("RUN_TAG", ""),
                   help="subdirectory under artifacts/rog_parametric/, so a second "
                        "concurrent instance does not overwrite this run")
    p.add_argument("--compare-dir", default=env("COMPARE_DIR", ""),
                   help="results dir of the KG-grounded cache experiment to ablate "
                        "against (default: artifacts/rog_cache/<run-tag>). Its rows "
                        "are read from summary.csv, else from its manifests.")
    p.add_argument("--reference", default=env("REFERENCE", "none"),
                   help="which KG-grounded policy the delta is taken against "
                        "(default: 'none', the uncached baseline, so the delta "
                        "isolates the KG rather than a cache policy)")
    args = p.parse_args()

    split = split_for(args.limit)
    out_host = OUT_HOST / args.run_tag if args.run_tag else OUT_HOST
    compare_dir = (Path(args.compare_dir) if args.compare_dir
                   else (CACHE_OUT_HOST / args.run_tag if args.run_tag else CACHE_OUT_HOST))
    if not compare_dir.is_absolute():
        compare_dir = (REPO_ROOT / compare_dir).resolve()

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys()
    load_dotenv(required=("HF_TOKEN",) if n_keys else ("HF_TOKEN", "LLM_API_KEY"))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; "
              f"the client falls back to the next one on a 400/5xx")
    docker_build()
    out_host.mkdir(parents=True, exist_ok=True)

    # The model this run will actually use, with the vendor preset filled in --
    # recorded, guarded and printed, so a tag can never again claim one model
    # while the run quietly uses another.
    model = resolve_model(args.vendor, args.model)
    warn_tag_model_mismatch(args.run_tag, model)
    guard_run_identity(out_host,
                       {"dataset": args.dataset, "limit": args.limit,
                        "vendor": args.vendor, "model": model, "seed": args.seed},
                       fresh=args.fresh)

    extra_env = {"LLM_DUMP_DIR": "/out"}
    if os.environ.get("LLM_API_KEY"):
        extra_env["LLM_API_KEY"] = os.environ["LLM_API_KEY"]
    if n_keys:
        # Mounted rather than passed as -e values: the container reads the same
        # file the host does, so the pool has exactly one parser.
        extra_env["LLM_KEYS_FILE"] = "/keys/.env_keys"

    home = os.path.expanduser("~")
    rog = make_rog_runner(
        gpus=False,  # nothing goes on the GPU: the reasoner is a remote API call
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

    # Upstream derives this directory from the absence of --add_rule; the name is
    # upstream's, and it is the on-disk evidence that no rules were used.
    predict_dir = out_host / "KGQA" / PARAMETRIC / args.dataset / MODEL_NAME / split / "no_rule"
    predict_file = f"/out/KGQA/{PARAMETRIC}/{args.dataset}/{MODEL_NAME}/{split}/no_rule/predictions.jsonl"

    done_marker = out_host / f"{PARAMETRIC}.done"
    if args.fresh:
        done_marker.unlink(missing_ok=True)

    print("\n" + "=" * 64)
    print(f">>> PARAMETRIC (no-KG) ablation  [{args.dataset}, {split}, "
          f"api:{args.vendor}, model={model}"
          f"{' (vendor default)' if not args.model else ''}]")
    print(">>> the model gets the question and nothing else: no paths, no "
          "triples, no retrieval")
    print("=" * 64)

    if done_marker.exists():
        print(f">>> already complete; skipping the run and re-reporting its scores. "
              f"Pass --fresh to redo.")
    else:
        print("\n>>> STAGE 1/2  reasoner (no KG, parametric prompt)   question -> answer")
        rog(["python", "/rogcache/predict_answer_parametric.py",
             "--model_name", MODEL_NAME, "-d", args.dataset, "--split", split,
             "--prompt_path", "prompts/llama2_predict.txt",
             "--vendor", args.vendor, *(["--model", args.model] if args.model else []),
             "--predict_path", f"/out/KGQA/{PARAMETRIC}", "--seed", args.seed,
             *(["--force"] if args.fresh else [])])

        print("\n>>> STAGE 2/2  scoring   (Hits@1 / F1)")
        rog(["python", "src/qa_prediction/evaluate_results.py",
             "-d", predict_file, "--cal_f1"])
        done_marker.write_text("")

    # Same manifest shape as the cache experiment's, minus the stage-1 keys it
    # has no stage 1 for, so the two runs' outputs can be read by one reader.
    manifest = {
        "tag": PARAMETRIC,
        "policy": PARAMETRIC,
        "kg": False,
        "dataset": args.dataset,
        "split": split,
        "vendor": args.vendor,
        "model": args.model or "",
        "eval_file": predict_file.replace("predictions.jsonl", "eval_result.txt"),
        "stage2_metrics": f"{predict_file}.metrics.jsonl",
        "predict_file": predict_file,
    }
    (out_host / f"manifest_{PARAMETRIC}.json").write_text(json.dumps(manifest, indent=2))

    row = {"condition": PARAMETRIC, "tag": PARAMETRIC}
    row.update(parse_eval_file(predict_dir / "eval_result.txt"))
    row.update(read_sidecar(predict_dir / "predictions.jsonl.metrics.jsonl"))

    kg_rows = load_kg_rows(compare_dir)
    if not kg_rows:
        print(f"\n[warn] no KG-grounded results under {compare_dir} -- run "
              f"scripts/run_rog_cache_experiment.py (or pass --compare-dir) to get "
              f"the comparison this ablation exists for")
    # Reference first, then the rest of the KG policies, then the no-KG row: the
    # table reads top-down from most knowledge available to least.
    kg_rows.sort(key=lambda r: (r.get("condition") != args.reference,
                                str(r.get("condition"))))
    emit_ablation(
        out_dir=out_host,
        rows=kg_rows + [row],
        columns=ABLATION_COLUMNS,
        subject=PARAMETRIC,
        reference=args.reference if any(r.get("condition") == args.reference
                                        for r in kg_rows) else None,
        title=f"ABLATION  RoG: no KG (parametric) vs KG-grounded  "
              f"[{args.dataset}, {split}, {args.vendor}]",
        count_key="n_questions",
        notes=(
            f"KG-grounded rows read from {compare_dir}",
            "scores are RoG's, on a 0-100 scale (evaluate_results.py)",
        ),
    )
    print(f"\n>>> raw outputs under {out_host.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
