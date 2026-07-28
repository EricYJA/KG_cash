"""Sweep the question-cache policies over ToG, scoring each with eval.py.

Runs one config per --policies entry (default: none exact semantic_lfu
semantic_lru semantic_oracle), mirroring the RoG experiment's policy sweep so the
two are directly comparable. 'none' is the uncached baseline; every other policy
runs the same single pass with that cache policy enabled (cold, its own cache
file). Each config's output JSONL is scored for Exact Match / Hits@1 / F1, and
the per-policy rows are written to summary.json (fed to summarize_tog_cache.py ->
plot_rog_cache_results.py). With --loop N (N>1) each policy instead runs
main_freebase_loop.py for N passes so the cache warms across passes.

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
    # Stream the child's output live (so long runs aren't silent) while still
    # capturing it -- eval_jsonl() parses the returned text. PYTHONUNBUFFERED
    # forces the child to flush promptly through the pipe (it's not a TTY).
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


def eval_jsonl(jsonl_path: Path, dataset: str) -> dict:
    """Run eval.py on a JSONL and parse its stdout into a metrics dict."""
    out = run(
        [sys.executable, "eval.py", "--dataset", dataset,
         "--output_file", str(jsonl_path)],
        cwd=EVAL_DIR,
    )

    def _num(pattern: str) -> float:
        m = re.search(pattern, out)
        return float(m.group(1)) if m else 0.0

    em = _num(r"Exact Match:\s*([0-9.]+)")
    right = error = 0
    rt_match = re.search(r"right:\s*(\d+),\s*error:\s*(\d+)", out)
    if rt_match:
        right = int(rt_match.group(1))
        error = int(rt_match.group(2))
    hits1 = _num(r"Hits@1:\s*([0-9.]+)")
    f1 = _num(r"F1:\s*([0-9.]+)")
    precision = _num(r"Precision:\s*([0-9.]+)")
    recall = _num(r"Recall:\s*([0-9.]+)")
    # Total = number of records in the JSONL (eval skips refusals so
    # right+error may be < total).
    with jsonl_path.open() as f:
        total = sum(1 for line in f if line.strip())
    return {"exact_match": em, "right": right, "error": error, "records": total,
            "hits1": hits1, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="webqsp")
    parser.add_argument("--test-limit", default="20",
                        help="how many samples per run (keeps token cost bounded), or 'all'. "
                             "Forwarded verbatim to main_freebase.py, which parses it.")
    parser.add_argument("--vendor", default="tamu")
    parser.add_argument("--model", default="",
                        help="override the vendor's default model id; forwarded to "
                             "main_freebase.py / main_freebase_loop.py.")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--capacity", type=int, default=4096,
                        help="max cached questions per policy (LRU/LFU eviction). "
                             "Forwarded to main_freebase.py as "
                             "--question-cache-capacity; ignored by the 'none' "
                             "baseline.")
    parser.add_argument("--policies",
                        default="semantic_lru semantic_lfu none exact semantic_oracle",
                        help="cache policies to sweep, in run order (space- or "
                             "comma-separated). 'none' is the uncached baseline; the "
                             "rest map to main_freebase.py --cache-policy. Covers the "
                             "same policies as the RoG experiment so the two are "
                             "comparable.")
    parser.add_argument("--loop", type=int, default=1,
                        help="passes per policy. 1 (default) = single pass via "
                             "main_freebase.py, matching RoG; >1 uses "
                             "main_freebase_loop.py to warm the cache.")
    parser.add_argument("--cache-dir", default=str(OUTPUT_DIR / "compare_caches"),
                        help="dir for per-config cache JSON files (cleared on start).")
    parser.add_argument("--results-dir", default=str(OUTPUT_DIR / "compare_results"),
                        help="dir for per-config JSONL output files.")
    parser.add_argument("--fresh", action="store_true",
                        help="wipe this tag's caches and outputs and start over. "
                             "Default resumes: completed configs are skipped and "
                             "partial ones continue where they stopped.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    # Resume by default: keep prior caches/outputs so an interrupted run continues
    # where it stopped (each config skips questions already in its JSONL, and the
    # semantic cache persists on disk). --fresh forces a clean slate.
    if args.fresh:
        for d in (cache_dir, results_dir):
            if d.exists():
                shutil.rmtree(d)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--dataset", args.dataset,
        "--test-limit", str(args.test_limit),
        "--vendor", args.vendor,
        "--depth", str(args.depth),
        "--width", str(args.width),
    ]
    if args.model:
        common += ["--model", args.model]

    # One config per cache policy, mirroring the RoG experiment's --policies sweep.
    # 'none' is the uncached baseline; every other policy runs the same single pass
    # with its cache enabled (cold, its own cache file). loop>1 switches to the
    # loop runner so the cache can warm across passes.
    policies = [p for p in args.policies.replace(",", " ").split() if p]
    looped = args.loop > 1
    runner = "main_freebase_loop.py" if looped else "main_freebase.py"

    configs: list[tuple[str, list[str], Path]] = []
    for policy in policies:
        out_path = results_dir / f"{policy}.jsonl"
        cmd = [sys.executable, runner, *common, "--output-file", str(out_path)]
        if looped:
            cmd += ["--loop", str(args.loop)]
        if policy == "none":
            cmd += ["--no-question-cache"]
        else:
            cmd += ["--question-cache-path", str(cache_dir / f"{policy}.json"),
                    "--cache-policy", policy,
                    "--similarity-threshold", str(args.similarity_threshold),
                    "--question-cache-capacity", str(args.capacity)]
        configs.append((policy, cmd, out_path))

    rows: list[dict] = []
    for name, cmd, out_path in configs:
        done_marker = results_dir / f"{name}.done"
        if done_marker.exists() and not args.fresh:
            print(f"\n[resume] policy {name!r} already complete; skipping its run "
                  f"(delete {done_marker} or pass --fresh to redo)")
        else:
            # No unlink: main_freebase.py / main_freebase_loop.py resume from an
            # existing JSONL, skipping questions already answered.
            run(cmd, cwd=HERE)
            done_marker.write_text("")  # mark complete only after the run succeeds
        metrics = eval_jsonl(out_path, args.dataset)
        rows.append({"config": name, "policy": name, **metrics, "output": str(out_path)})

    print("\n" + "=" * 78)
    print(f"{'policy':<16} {'records':>8} {'right':>6} {'error':>6} "
          f"{'EM':>8} {'Hits@1':>8} {'F1':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['policy']:<16} {r['records']:>8} {r['right']:>6} "
              f"{r['error']:>6} {r['exact_match']:>8.4f} {r['hits1']:>8.4f} {r['f1']:>8.4f}")
    print("=" * 78)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
