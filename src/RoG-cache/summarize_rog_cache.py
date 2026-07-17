"""Collect one row per cache policy from a RoG cache-experiment run.

Reads every `manifest_<tag>.json` that scripts/run_rog_cache_experiment.py drops
in the results dir. Each manifest points at that policy's two outputs:

    cache_stats  ->  hit rate, LLM calls saved, wall time   (stage 1)
    eval_file    ->  Accuracy / Hit / F1 / Precision / Recall (stage 3)

and this joins them into `summary.json` (full records) and `summary.csv` (the
columns you actually read), plus a table on stdout.

The point of the join: hit rate alone cannot tell you whether caching *hurt*.
Reusing another question's relation paths can quietly cost accuracy, so the
cache row and the score row have to be read together.

Usage (inside the rog-eval image):
    python summarize_rog_cache.py --results-dir /out
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Order matters: this is the table people read, cache behaviour then accuracy.
CSV_COLUMNS = [
    "policy", "tag", "hit_rate", "n_questions",
    "planner_llm_calls", "planner_llm_calls_saved",
    "hits", "misses", "hit_total_s", "miss_total_s", "avg_hit_s", "avg_miss_s",
    "estimated_time_saved_s", "speedup_x", "wall_s_total",
    "accuracy", "hit", "f1", "precision", "recall",
]

METRIC_RE = re.compile(
    r"Accuracy:\s*([\d.]+)\s+Hit:\s*([\d.]+)\s+F1:\s*([\d.]+)\s+"
    r"Precision:\s*([\d.]+)\s+Recall:\s*([\d.]+)"
)


def resolve(path_str, results_dir):
    """Resolve a manifest path, tolerating a different mount point.

    Manifests written in-container hold container-absolute paths (/out/...). If
    the summary is regenerated on the host, /out no longer exists, so fall back
    to re-rooting the tail of the path onto results_dir.
    """
    path = Path(path_str)
    if path.exists():
        return path
    parts = path.parts
    for i in range(len(parts)):
        candidate = results_dir.joinpath(*parts[i:])
        if candidate.exists():
            return candidate
    return path  # let the caller report it as missing


def parse_eval_file(path):
    """Pull the five metrics out of RoG's eval_result.txt, or {} if unreadable."""
    if not path.exists():
        return {}
    match = METRIC_RE.search(path.read_text())
    if not match:
        return {}
    accuracy, hit, f1, precision, recall = (float(g) for g in match.groups())
    return {
        "accuracy": accuracy, "hit": hit, "f1": f1,
        "precision": precision, "recall": recall,
    }


def build_row(manifest_path, results_dir):
    manifest = json.loads(manifest_path.read_text())
    row = {"tag": manifest["tag"], "policy": manifest["policy"]}

    stats_path = resolve(manifest["cache_stats"], results_dir)
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        timing = stats.pop("timing", {}) or {}
        row.update(timing)  # hoist hits/misses/avg_* to the top level
        row.update(stats)
        # The planner records policy "off" when caching is disabled; the manifest's
        # name for the same run is "none". The manifest wins.
        row["policy"] = manifest["policy"]
    else:
        print(f"  [warn] missing cache stats: {manifest['cache_stats']}")

    eval_path = resolve(manifest["eval_file"], results_dir)
    metrics = parse_eval_file(eval_path)
    if not metrics:
        print(f"  [warn] no metrics parsed from: {manifest['eval_file']}")
    row.update(metrics)
    return row


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results-dir", default="/out")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    manifests = sorted(results_dir.glob("manifest_*.json"))
    if not manifests:
        raise SystemExit(f"no manifest_*.json under {results_dir}")

    rows = [build_row(m, results_dir) for m in manifests]
    # Uncached baseline first so the deltas below it read naturally.
    rows.sort(key=lambda r: (r.get("policy") != "none", r.get("tag") or ""))

    (results_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    with (results_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    header = f"{'policy':<22}{'hit_rate':>10}{'calls':>8}{'saved':>8}{'wall_s':>10}{'Hits@1':>9}{'F1':>8}"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(key, spec):
            value = row.get(key)
            return format(value, spec) if isinstance(value, (int, float)) else "-"
        print(
            f"{str(row.get('tag')):<22}"
            f"{fmt('hit_rate', '>10.3f')}"
            f"{fmt('planner_llm_calls', '>8')}"
            f"{fmt('planner_llm_calls_saved', '>8')}"
            f"{fmt('wall_s_total', '>10.1f')}"
            f"{fmt('hit', '>9.1f')}"
            f"{fmt('f1', '>8.1f')}"
        )
    print(f"\nwrote {results_dir/'summary.csv'} and {results_dir/'summary.json'}")


if __name__ == "__main__":
    main()
