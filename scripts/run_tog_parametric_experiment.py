#!/usr/bin/env python3
"""ToG with the knowledge graph removed: the parametric-knowledge ablation.

Same workflow as scripts/run_tog_cache_experiment.py -- same dataset, same
samples, same vendor, same eval.py scoring (Exact Match / Hits@1 / F1) -- with
the knowledge graph taken out. main_freebase_parametric.py answers each question
in one LLM call: no SPARQL traversal, no relation/entity scoring calls, no
reasoning chain, and so no question cache either (a chain cache has nothing to
store when no chain is built).

Nothing from the KG reaches the model, and the prompt is changed to license the
model's own parametric knowledge -- ToG's prompts otherwise instruct it to answer
"with these triplets", which is meaningless when there are none. Grounding stays
off in every other sense: the client posts a plain chat-completions payload with
no tools, no retrieval and no vendor search-grounding, so the weights are the
only knowledge source.

The point is the ABLATION printed at the end: this run's scores next to the cache
experiment's KG-grounded policies, and the delta between them. That delta is what
the knowledge graph is worth on this split -- the part of the grounded score the
model could not have produced from memory alone.

Requires the KG_cash conda env (override with --conda-env / CONDA_ENV) and an
LLM key in .env / .env_keys. It does NOT need Virtuoso or Oxigraph, and does not
start one: there is nothing to query.

    ./scripts/run_tog_parametric_experiment.py
    ./scripts/run_tog_parametric_experiment.py --dataset cwq --limit 50
    ./scripts/run_tog_parametric_experiment.py --vendor google --limit all
    ./scripts/run_tog_parametric_experiment.py --compare-dir src/ToG-cache/output/compare_results/oxigraph

Outputs land under src/ToG-cache/output/parametric_results/<run-tag>/, never in
the cache experiment's compare_results/, so the two cannot overwrite each other.

Restartable: re-using a --run-tag RESUMES (main_freebase_parametric.py skips
questions already in its JSONL). --fresh (env FRESH=1) wipes the tag and redoes it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from _ablation import (emit_ablation, guard_run_identity,  # noqa: E402
                       resolve_model, warn_tag_model_mismatch)
from _tog_common import (  # noqa: E402
    EVAL_DIR, KEYS_PATH, REPO_ROOT, TOG_DIR, load_dotenv, load_env_keys,
    add_env_file_arg, preload_dotenv, python_cmd, run_py,
)

OUTPUT_DIR = REPO_ROOT / "src" / "ToG-cache" / "output"
PARAMETRIC = "parametric_no_kg"

# (row key, header, format). ToG's eval.py reports on a 0-1 scale.
ABLATION_COLUMNS = [
    ("records", "n", ">8"),
    ("hits1", "Hits@1", ">10.4f"),
    ("f1", "F1", ">9.4f"),
    ("precision", "Prec", ">9.4f"),
    ("recall", "Recall", ">9.4f"),
    ("exact_match", "EM", ">9.4f"),
    ("llm_calls", "calls", ">8"),
    ("wall_s_total", "wall_s", ">10.1f"),
]


def run_capture(cmd: list[str], cwd: Path) -> str:
    """Run a command, streaming its output live and returning it as text.

    Streamed because a full split takes a while and a silent run is
    indistinguishable from a hung one; captured because eval.py reports its
    metrics on stdout and there is nowhere else to read them from.
    """
    print(f"\n[run] (cwd={cwd}) {' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=cwd, text=True, bufsize=1, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return "".join(captured)


def eval_jsonl(jsonl_path: Path, dataset: str, conda_env: str) -> dict:
    """Score a results JSONL with ToG's eval.py and parse its stdout.

    Same scorer and same parsing as compare_cache_accuracy.py's, so the numbers
    below the ablation table are produced by the code that produced the ones
    above it.
    """
    out = run_capture(
        python_cmd(conda_env) + ["eval.py", "--dataset", dataset,
                                 "--output_file", str(jsonl_path)],
        cwd=EVAL_DIR,
    )

    def num(pattern: str) -> float:
        m = re.search(pattern, out)
        return float(m.group(1)) if m else 0.0

    right = error = 0
    rt = re.search(r"right:\s*(\d+),\s*error:\s*(\d+)", out)
    if rt:
        right, error = int(rt.group(1)), int(rt.group(2))
    with jsonl_path.open() as f:
        total = sum(1 for line in f if line.strip())
    return {"exact_match": num(r"Exact Match:\s*([0-9.]+)"),
            "right": right, "error": error, "records": total,
            "hits1": num(r"Hits@1:\s*([0-9.]+)"), "f1": num(r"F1:\s*([0-9.]+)"),
            "precision": num(r"Precision:\s*([0-9.]+)"),
            "recall": num(r"Recall:\s*([0-9.]+)")}


def read_sidecar(path: Path) -> dict:
    """Sum the per-question metrics sidecar (stdlib; cache_metrics' reader needs utils)."""
    if not path.exists():
        return {}
    n = calls = 0
    elapsed = 0.0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        elapsed += rec.get("elapsed_s", 0.0)
        calls += rec.get("llm_calls", 0)
    return {"llm_calls": calls, "wall_s_total": round(elapsed, 3), "n_questions": n}


def load_kg_rows(compare_dir: Path, dataset: str, conda_env: str) -> list[dict]:
    """One row per KG-grounded policy from a cache-experiment run.

    Prefers summary.json, which compare_cache_accuracy.py writes at the end of a
    sweep with exactly the metric keys used above. A sweep that was interrupted
    before that step still has its per-policy JSONLs, so those are scored here
    instead -- an unfinished sweep should contribute the policies it did finish,
    not nothing at all.

    Either way each policy's metrics sidecar supplies its wall time and LLM
    calls. Those are whole-question numbers, as the parametric run's are, so the
    two sit in one column honestly.
    """
    def with_cost(row: dict, output: str | None) -> dict:
        if output:
            sidecar = Path(f"{output}.metrics.jsonl")
            if not sidecar.exists():  # summary written against another mount
                sidecar = compare_dir / f"{Path(output).name}.metrics.jsonl"
            row.update(read_sidecar(sidecar))
        return row

    summary_path = compare_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        return [with_cost({"condition": raw.get("policy") or raw.get("config"), **raw},
                          raw.get("output"))
                for raw in payload.get("rows", [])]

    if not compare_dir.is_dir():
        return []
    rows = []
    for jsonl in sorted(compare_dir.glob("*.jsonl")):
        if jsonl.name.endswith(".metrics.jsonl"):
            continue
        print(f"\n[compare] no summary.json under {compare_dir}; scoring "
              f"{jsonl.name} directly")
        rows.append(with_cost({"condition": jsonl.stem,
                               **eval_jsonl(jsonl, dataset, conda_env)}, str(jsonl)))
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
    p.add_argument("--conda-env", default=env("CONDA_ENV", "KG_cash"))
    p.add_argument("--dataset", default=env("DATASET", "webqsp"),
                   help="webqsp | cwq | qald | ...")
    p.add_argument("-n", "--limit", default=env("N", "20"),
                   help="samples to run, or 'all'")
    p.add_argument("--vendor", default=env("VENDOR", "tamu"), help="tamu | openai | google")
    p.add_argument("--model", default=env("MODEL", ""),
                   help="override the vendor's default model id")
    p.add_argument("--run-tag", default=env("RUN_TAG", ""),
                   help="subdirectory under output/parametric_results/, so a second "
                        "concurrent instance does not overwrite this run. Reusing the "
                        "cache experiment's tag also pairs the two automatically, "
                        "since --compare-dir defaults to the same tag.")
    p.add_argument("--fresh", action="store_true", default=env("FRESH", "0") == "1",
                   help="wipe this run-tag's outputs and start over. Default resumes.")
    p.add_argument("--compare-dir", default=env("COMPARE_DIR", ""),
                   help="results dir of the KG-grounded cache experiment to ablate "
                        "against (default: output/compare_results/<run-tag>, else "
                        "output/compare_results/virtuoso -- the tag "
                        "run_tog_cache_experiment.py uses when given none)")
    p.add_argument("--reference", default=env("REFERENCE", "none"),
                   help="which KG-grounded policy the delta is taken against "
                        "(default: 'none', the uncached baseline, so the delta "
                        "isolates the KG rather than a cache policy)")
    args = p.parse_args()

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys()
    load_dotenv(required=() if n_keys else ("LLM_API_KEY",))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; "
              f"the client falls back to the next one on a 400/5xx")

    results_root = OUTPUT_DIR / "parametric_results"
    results_dir = results_root / args.run_tag if args.run_tag else results_root
    if args.fresh and results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = results_dir / f"{PARAMETRIC}.jsonl"

    # The model this run will actually use, with the vendor preset filled in --
    # recorded, guarded and printed, so a tag can never again claim one model
    # while the run quietly uses another.
    model = resolve_model(args.vendor, args.model)
    warn_tag_model_mismatch(args.run_tag, model)
    guard_run_identity(results_dir,
                       {"dataset": args.dataset, "test_limit": args.limit,
                        "vendor": args.vendor, "model": model},
                       fresh=args.fresh)

    # A shared tag pairs this ablation with the cache run it belongs to; with no
    # tag, fall back to the tag that runner picks for itself by default.
    compare_dir = Path(args.compare_dir) if args.compare_dir else \
        OUTPUT_DIR / "compare_results" / (args.run_tag or "virtuoso")
    if not compare_dir.is_absolute():
        compare_dir = (REPO_ROOT / compare_dir).resolve()

    print("\n" + "=" * 64)
    print(f">>> ToG parametric (no-KG) ablation  [dataset={args.dataset}, "
          f"N={args.limit}, vendor={args.vendor},")
    print(f">>>   model={model}"
          f"{' (vendor default)' if not args.model else ''}, "
          f"no SPARQL backend, no cache]")
    print(">>> the model gets the question and nothing else: no triples, no "
          "traversal, no retrieval")
    print("=" * 64)

    done_marker = results_dir / f"{PARAMETRIC}.done"
    if done_marker.exists():
        print(f">>> already complete; skipping the run and re-scoring its output. "
              f"Pass --fresh to redo.")
    else:
        run_py(
            [
                "main_freebase_parametric.py",
                "--dataset", args.dataset,
                "--test-limit", args.limit,
                "--vendor", args.vendor,
                *(["--model", args.model] if args.model else []),
                "--output-file", str(out_jsonl),
            ],
            cwd=TOG_DIR,
            conda_env=args.conda_env,
        )
        done_marker.write_text("")  # only after the run succeeds

    row = {"condition": PARAMETRIC,
           **eval_jsonl(out_jsonl, args.dataset, args.conda_env),
           **read_sidecar(Path(f"{out_jsonl}.metrics.jsonl"))}

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(
        {"args": vars(args), "rows": [{**row, "output": str(out_jsonl)}]}, indent=2))
    print(f"\nWrote {summary_path}")

    kg_rows = load_kg_rows(compare_dir, args.dataset, args.conda_env)
    if not kg_rows:
        print(f"\n[warn] no KG-grounded results under {compare_dir} -- run "
              f"scripts/run_tog_cache_experiment.py (or pass --compare-dir) to get "
              f"the comparison this ablation exists for")
    # Reference first, then the other KG policies, then the no-KG row: the table
    # reads top-down from most knowledge available to least.
    kg_rows.sort(key=lambda r: (r.get("condition") != args.reference,
                                str(r.get("condition"))))
    emit_ablation(
        out_dir=results_dir,
        rows=kg_rows + [row],
        columns=ABLATION_COLUMNS,
        subject=PARAMETRIC,
        reference=args.reference if any(r.get("condition") == args.reference
                                        for r in kg_rows) else None,
        title=f"ABLATION  ToG: no KG (parametric) vs KG-grounded  "
              f"[{args.dataset}, N={args.limit}, {args.vendor}]",
        count_key="records",
        notes=(
            f"KG-grounded rows read from {compare_dir}",
            "scores are eval.py's, on a 0-1 scale",
        ),
    )


if __name__ == "__main__":
    main()
