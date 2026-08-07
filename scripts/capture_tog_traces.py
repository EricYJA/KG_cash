#!/usr/bin/env python3
"""Capture a full ToG KG/LLM trace in parallel, then merge it back into one file.

cache_simulator.py replays output/traces/tog_trace_<dataset>.json. Regenerating
that file means re-running ToG over the whole split, which is ~99.5% LLM wait:
on the existing traces a question costs 3.8 LLM calls at 7.6s (WebQSP) or 3.9 at
10.5s (CWQ), against 57s of KG time for the entire 400-question run. Sequentially
that is ~13h for WebQSP and ~40h for CWQ.

Questions are independent, so this shards them across N processes and cuts wall
time to roughly 1/N of that. Processes, not threads: TraceRecorder keeps the
in-flight question's events in instance attributes and trace_instrument uses
module-global label queues, so two questions running in one interpreter would
interleave their events into a single buffer and corrupt both traces.

Shards are strided (datas[i::N]) rather than contiguous, because question cost
varies by more than an order of magnitude and a contiguous split leaves one
worker running long after the others have finished.

Merging restores the dataset's own question order. That matters: the simulator's
"sequential" access pattern is the real order questions arrived in, and the
LRU/LFU hit rates depend on it. Shard files are concatenated and then sorted
against the dataset, not just appended.

Usage:
    # measure the LLM's tolerated concurrency first
    scripts/capture_tog_traces.py --dataset webqsp --test-limit 32 --workers 4 --probe

    # the real thing
    scripts/capture_tog_traces.py --dataset webqsp --workers 16
    scripts/capture_tog_traces.py --dataset cwq --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOG_DIR = REPO_ROOT / "src" / "ToG-cache" / "ToG"
TRACE_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "traces"
# The runner needs rank_bm25 / sentence_transformers / openai, which only the
# KG_cash env has; plain `python3` on this box cannot import utils.py.
DEFAULT_PYTHON = "/home/stanic/anaconda3/envs/KG_cash/bin/python"


def shard_paths(work_dir: Path, dataset: str, index: int) -> tuple[Path, Path]:
    return (
        work_dir / f"{dataset}_shard{index}_answers.jsonl",
        work_dir / f"{dataset}_shard{index}_trace.jsonl",
    )


def launch(args, work_dir: Path) -> list[subprocess.Popen]:
    procs = []
    for index in range(args.workers):
        answers, trace = shard_paths(work_dir, args.dataset, index)
        cmd = [
            args.python, "main_freebase.py",
            "--dataset", args.dataset,
            "--vendor", args.vendor,
            "--no-question-cache",          # every question must do a full traversal
            "--timing-log", "",             # per-run aggregate is meaningless per shard
            "--output-file", str(answers),
            "--trace-output", str(trace),
            "--shard-index", str(index),
            "--shard-count", str(args.workers),
        ]
        if args.test_limit is not None:
            cmd += ["--test-limit", str(args.test_limit)]
        if args.model:
            cmd += ["--model", args.model]
        log = open(work_dir / f"{args.dataset}_shard{index}.log", "a")
        procs.append(subprocess.Popen(cmd, cwd=TOG_DIR, stdout=log, stderr=subprocess.STDOUT))
    return procs


def wait_for(procs: list[subprocess.Popen], work_dir: Path, dataset: str) -> list[int]:
    failed = []
    start = time.time()
    while True:
        alive = [p for p in procs if p.poll() is None]
        done = len(procs) - len(alive)
        traced = sum(
            sum(1 for _ in open(t))
            for t in work_dir.glob(f"{dataset}_shard*_trace.jsonl")
        )
        print(f"\r[{time.time() - start:7.0f}s] shards done {done}/{len(procs)} | "
              f"questions traced {traced}", end="", flush=True)
        if not alive:
            break
        time.sleep(10)
    print()
    for index, proc in enumerate(procs):
        if proc.returncode != 0:
            failed.append(index)
            print(f"[capture] shard {index} exited {proc.returncode}; "
                  f"see {work_dir / f'{dataset}_shard{index}.log'}", file=sys.stderr)
    return failed


def dataset_order(dataset: str, python: str) -> list[str]:
    """The dataset's questions in their natural order, via ToG's own loader."""
    code = (
        "import json;from utils import prepare_dataset;"
        "d,q=prepare_dataset(%r);print(json.dumps([x[q] for x in d]))" % dataset
    )
    out = subprocess.run(
        [python, "-c", code], cwd=TOG_DIR, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def merge(work_dir: Path, dataset: str, order: list[str], out_jsonl: Path) -> int:
    traces = []
    for shard in sorted(work_dir.glob(f"{dataset}_shard*_trace.jsonl")):
        with open(shard, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    traces.append(json.loads(line))

    # Restore dataset order. Duplicate questions keep their relative shard order.
    rank = {question.strip(): i for i, question in enumerate(order)}
    unknown = [t for t in traces if t.get("question", "").strip() not in rank]
    if unknown:
        print(f"[capture] {len(unknown)} traced questions not found in the dataset "
              f"(kept, appended last)", file=sys.stderr)
    traces.sort(key=lambda t: rank.get(t.get("question", "").strip(), len(rank)))

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    pretty = out_jsonl.with_suffix(".json")
    with pretty.open("w", encoding="utf-8") as handle:
        json.dump(traces, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return len(traces)


def summarize(out_jsonl: Path) -> None:
    import collections

    traces = [json.loads(l) for l in open(out_jsonl, encoding="utf-8") if l.strip()]
    counts = collections.Counter(
        (e["type"], e["operation"]) for t in traces for e in t["events"]
    )
    kg = [e for t in traces for e in t["events"] if e["type"] == "KG"]
    keys = {
        json.dumps({"operation": e["operation"], "input": e.get("input", {})}, sort_keys=True)
        for e in kg
    }
    print(f"\n[capture] {len(traces)} questions, {len(kg)} KG requests, "
          f"{len(keys)} unique keys "
          f"(reuse ceiling {100 * (1 - len(keys) / max(len(kg), 1)):.2f}%)")
    for (etype, op), n in sorted(counts.items()):
        print(f"    {etype:<6} {op:<34} {n}")
    missing = {"relation_lookup_head", "relation_lookup_tail",
               "entity_search", "entity_name_resolve"} - {e["operation"] for e in kg}
    if missing:
        print(f"[capture] WARNING: no {', '.join(sorted(missing))} events. "
              f"The SPARQL endpoint is probably not serving Freebase.", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test-limit", type=int, default=None,
                    help="only the first N questions (before sharding)")
    ap.add_argument("--vendor", default="tamu")
    ap.add_argument("--model", default="")
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "artifacts" / "trace_capture")
    ap.add_argument("--output", type=Path, default=None,
                    help="merged trace .jsonl (default: output/traces/tog_trace_<dataset>.jsonl)")
    ap.add_argument("--probe", action="store_true",
                    help="throughput probe: report questions/hour and stop before merging")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe shard files first; default resumes them")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip the run, just merge whatever shard files exist")
    args = ap.parse_args()

    work_dir = args.work_dir / args.dataset
    if args.fresh and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output or (TRACE_DIR / f"tog_trace_{args.dataset}.jsonl")

    if not args.merge_only:
        endpoint = os.environ.get("SPARQL_ENDPOINT", "http://localhost:8890/sparql")
        print(f"[capture] {args.dataset}: {args.workers} shards, endpoint {endpoint}")
        started = time.time()
        procs = launch(args, work_dir)
        failed = wait_for(procs, work_dir, args.dataset)
        elapsed = time.time() - started
        traced = sum(sum(1 for _ in open(t))
                     for t in work_dir.glob(f"{args.dataset}_shard*_trace.jsonl"))
        rate = traced / elapsed * 3600 if elapsed else 0
        print(f"[capture] {traced} questions in {elapsed / 60:.1f} min "
              f"= {rate:.0f} questions/hour at {args.workers} workers")
        if args.probe:
            print(f"[capture] probe only; not merging. At this rate a full split of "
                  f"1639 (webqsp) would take {1639 / max(rate, 1e-9):.1f}h, "
                  f"3531 (cwq) {3531 / max(rate, 1e-9):.1f}h.")
            return
        if failed:
            print(f"[capture] {len(failed)} shard(s) failed; merging anyway, but the "
                  f"trace is incomplete. Re-run to resume.", file=sys.stderr)

    order = dataset_order(args.dataset, args.python)
    n = merge(work_dir, args.dataset, order, out_jsonl)
    print(f"[capture] merged {n} traces -> {out_jsonl} (+ .json)")
    summarize(out_jsonl)


if __name__ == "__main__":
    main()
