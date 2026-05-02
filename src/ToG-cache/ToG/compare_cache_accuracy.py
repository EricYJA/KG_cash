"""Run main_freebase.py and main_freebase_loop.py with and without the
question cache, then print a side-by-side Exact Match table.

Four configurations:
  1. main_freebase.py            --no-question-cache    (baseline_main)
  2. main_freebase.py            cache enabled          (cache_main)
  3. main_freebase_loop.py       --no-question-cache    (baseline_loop)
  4. main_freebase_loop.py       cache enabled          (cache_loop)

For (3) and (4), the loop is run twice over the same questions so the cache
can warm up (cache_loop's second pass should be near-100% hit). The eval
script is invoked on each output JSONL to compute Exact Match.

Run from src/ToG-cache/ToG/ (so eval.py's relative paths resolve).
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

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "eval"
OUTPUT_DIR = HERE.parent / "output"


def run(cmd: list[str], cwd: Path) -> str:
    print(f"\n[cmd] (cwd={cwd}) {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, check=False, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}): {' '.join(cmd)}")
    return proc.stdout


def eval_jsonl(jsonl_path: Path, dataset: str) -> tuple[float, int, int, int]:
    """Run eval.py on a JSONL and parse its stdout."""
    out = run(
        [sys.executable, "eval.py", "--dataset", dataset,
         "--output_file", str(jsonl_path)],
        cwd=EVAL_DIR,
    )
    em = right = error = total = 0
    em_match = re.search(r"Exact Match:\s*([0-9.]+)", out)
    rt_match = re.search(r"right:\s*(\d+),\s*error:\s*(\d+)", out)
    if em_match:
        em = float(em_match.group(1))
    if rt_match:
        right = int(rt_match.group(1))
        error = int(rt_match.group(2))
    # Total = number of records in the JSONL (eval skips refusals so
    # right+error may be < total).
    with jsonl_path.open() as f:
        total = sum(1 for line in f if line.strip())
    return em, right, error, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="webqsp")
    parser.add_argument("--test-limit", type=int, default=20,
                        help="how many samples per run (keeps token cost bounded).")
    parser.add_argument("--vendor", default="tamu")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--loop", type=int, default=2,
                        help="loop count for main_freebase_loop.py.")
    parser.add_argument("--cache-dir", default=str(OUTPUT_DIR / "compare_caches"),
                        help="dir for per-config cache JSON files (cleared on start).")
    parser.add_argument("--results-dir", default=str(OUTPUT_DIR / "compare_results"),
                        help="dir for per-config JSONL output files.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--dataset", args.dataset,
        "--test-limit", str(args.test_limit),
        "--vendor", args.vendor,
        "--depth", str(args.depth),
        "--width", str(args.width),
    ]

    configs: list[tuple[str, list[str], Path]] = []

    # 1. main, no cache
    p1 = results_dir / "baseline_main.jsonl"
    configs.append((
        "baseline_main",
        [sys.executable, "main_freebase.py", *common, "--no-question-cache",
         "--output-file", str(p1)],
        p1,
    ))

    # 2. main, with cache
    p2 = results_dir / "cache_main.jsonl"
    configs.append((
        "cache_main",
        [sys.executable, "main_freebase.py", *common,
         "--question-cache-path", str(cache_dir / "main.json"),
         "--similarity-threshold", str(args.similarity_threshold),
         "--output-file", str(p2)],
        p2,
    ))

    # 3. loop, no cache
    p3 = results_dir / "baseline_loop.jsonl"
    configs.append((
        "baseline_loop",
        [sys.executable, "main_freebase_loop.py", *common, "--no-question-cache",
         "--loop", str(args.loop),
         "--output-file", str(p3)],
        p3,
    ))

    # 4. loop, with cache
    p4 = results_dir / "cache_loop.jsonl"
    configs.append((
        "cache_loop",
        [sys.executable, "main_freebase_loop.py", *common,
         "--question-cache-path", str(cache_dir / "loop.json"),
         "--similarity-threshold", str(args.similarity_threshold),
         "--loop", str(args.loop),
         "--output-file", str(p4)],
        p4,
    ))

    rows: list[dict] = []
    for name, cmd, out_path in configs:
        if out_path.exists():
            out_path.unlink()
        run(cmd, cwd=HERE)
        em, right, error, total = eval_jsonl(out_path, args.dataset)
        rows.append({"config": name, "exact_match": em, "right": right,
                     "error": error, "records": total, "output": str(out_path)})

    print("\n" + "=" * 78)
    print(f"{'config':<16} {'records':>8} {'right':>6} {'error':>6} {'EM':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['config']:<16} {r['records']:>8} {r['right']:>6} "
              f"{r['error']:>6} {r['exact_match']:>8.4f}")
    print("=" * 78)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
