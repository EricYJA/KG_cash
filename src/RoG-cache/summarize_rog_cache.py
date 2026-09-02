"""Collect one row per cache policy from a RoG cache-experiment run.

Reads every `manifest_<tag>.json` that scripts/run_rog_cache_experiment.py drops
in the results dir. Each manifest points at that policy's outputs:

    cache_stats     ->  hit rate, LLM calls saved, wall time     (stage 1)
    stage1_metrics  ->  per-question planner times               (stage 1)
    stage2_metrics  ->  per-question reasoner times              (stage 2)
    eval_file       ->  Accuracy / Hit / F1 / Precision / Recall (stage 3)

and this joins them into `summary.json` (full records) and `summary.csv` (the
columns you actually read), plus a table on stdout.

The point of the join: hit rate alone cannot tell you whether caching *hurt*.
Reusing another question's relation paths can quietly cost accuracy, so the
cache row and the score row have to be read together.

Four speedup columns, in two pairs, and the distinction is the whole point of
reading this file. Bare names are the whole question; planner_* is stage 1 alone:

    full_speedup_x          whole run, stage 1 + stage 2: what every question
                            would have cost uncached over what the run cost.
                            Computed by ToG's aggregate_run_metrics over joined
                            per-question times, so it means exactly what ToG's
                            column of that name means. THIS is the system result.
    speedup_x               per question, stage 1 + stage 2: a cold question over
                            a cache-served one. Conditional on a hit, so it reads
                            far larger than full_speedup_x and is not a system
                            result -- at a 7.7% hit rate a 2.5x per-hit saving is
                            a 1.05x run.
    planner_full_speedup_x  the same whole-run figure over stage 1 alone
    planner_speedup_x       the same per-question figure over stage 1 alone

The planner_* pair attributes where a saving came from; neither is a system-level
number, because the reasoner call they exclude runs on every question, cached or
not, and bounds how much any planner cache can help.

Every column above is rebuilt from the per-question sidecars, never from a
process's end-of-run counters, so a policy that was interrupted and resumed
reports the whole split rather than whatever its last invocation happened to
handle. Runs made before the stage-2 sidecar existed have no stage-2 times to
join, so their end-to-end columns come out empty rather than being back-filled
from the planner-only number -- that substitution is the error this file exists
to undo.

Usage (inside the rog-eval image):
    python summarize_rog_cache.py --results-dir /out
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from rog_e2e_metrics import (
    aggregate_end_to_end,
    aggregate_run_metrics,
    metrics_sidecar_path,
)

# Order matters: this is the table people read, cache behaviour then accuracy.
# The end-to-end columns come first among the timings: they are the system-level
# result, and the planner_* ones beside them are the stage-1 attribution.
CSV_COLUMNS = [
    "policy", "tag", "hit_rate", "n_questions",
    "planner_llm_calls", "planner_llm_calls_saved",
    "hits", "misses",
    "hit_total_s", "miss_total_s", "avg_hit_s", "avg_miss_s",
    "estimated_time_saved_s", "full_speedup_x", "speedup_x", "wall_s_total",
    "planner_avg_hit_s", "planner_avg_miss_s",
    "planner_full_speedup_x", "planner_speedup_x",
    "accuracy", "hit", "f1", "precision", "recall",
]

# Every field of a stage-1 timing block is renamed planner_* on the way into the
# row, unless it already says planner_. The bare names belong to the whole-question
# numbers joined in from stage 2, so both can sit in one row with neither readable
# as the other.
#
# Prefixing everything rather than listing the fields keeps that true as
# cache_metrics.aggregate_run_metrics grows fields -- it has gained `failed`,
# `timed_hits`, `baseline_miss_s` and more since this was written. A whitelist
# silently drops each new field into the bare namespace, where it reads as an
# end-to-end number, which is the exact confusion this exists to prevent.
def planner_key(key):
    return key if key.startswith("planner_") else f"planner_{key}"


# Dropped from a stage-1 timing block rather than renamed. Stage 1 used to write
# `speedup_x` as its whole-run WALL-CLOCK ratio: a different quantity from the
# per-question ratio that name means everywhere else, taken over one process's
# clock (so wrong after a resume) and over work the per-question timer excludes.
# It was always overwritten by the end-to-end value and never reached a reader.
# Runs written after this emit `planner_speedup_x` instead; renaming the legacy
# key would put two meanings in one column depending on how old a run is.
LEGACY_STAGE1_TIMING_FIELDS = {"speedup_x"}

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


def stage_metrics_paths(manifest, stats, results_dir):
    """(stage 1, stage 2) sidecar paths for a policy, or None where unavailable.

    Prefers the manifest, which the experiment script writes explicitly. Falls
    back to deriving stage 1's from the prediction file recorded in cache_stats,
    so a run whose manifest predates those keys still joins.
    """
    stage1 = manifest.get("stage1_metrics") or stats.get("stage1_metrics_file")
    if not stage1 and stats.get("prediction_file"):
        stage1 = metrics_sidecar_path(stats["prediction_file"])
    stage2 = manifest.get("stage2_metrics")
    if not stage2 and manifest.get("predict_file"):
        stage2 = metrics_sidecar_path(manifest["predict_file"])

    resolved = []
    for path_str in (stage1, stage2):
        if not path_str:
            resolved.append(None)
            continue
        path = resolve(path_str, results_dir)
        resolved.append(path if path.exists() else None)
    return tuple(resolved)


def add_planner_timing(row, stage1_path):
    """Rebuild the planner_* columns from stage 1's per-question sidecar.

    cache_stats.json is written from counters that live in one process, so a
    policy that was interrupted and resumed described only the questions its last
    invocation handled -- the wrong n_questions, hit_rate and LLM-call count, on a
    run the resume machinery is otherwise careful to keep whole. The sidecar holds
    every question ever recorded for that output, so rebuilding from it is correct
    after any number of restarts, and it repairs runs summarized before
    gen_rule_path_api.py started writing restart-safe stats: their sidecars are
    already on disk, so re-running this file is enough.

    Also sets the bare hit_rate / n_questions / wall_s_total, which stage 1 can
    define alone (it sees every question, and its hit/miss labels the whole
    question). add_end_to_end_timing overwrites them with the joined values where
    stage 2's sidecar exists; where it does not, these are at least restart-safe
    planner numbers rather than process-local ones.
    """
    if not stage1_path:
        return
    timing, summary, _breakdown, _per_loop = aggregate_run_metrics(str(stage1_path))
    if not summary.get("n_questions"):
        return  # no records: leave whatever cache_stats.json supplied
    for key, value in timing.items():
        row[planner_key(key)] = value
    row["planner_llm_calls"] = summary["llm_calls"]
    # Exactly one planner call per question, so a hit saves exactly one -- no
    # estimator needed (ToG has to average, a ToG miss costing a variable number).
    row["planner_llm_calls_saved"] = timing["hits"]
    row["n_questions"] = summary["n_questions"]
    row["hit_rate"] = summary["hit_rate"]
    row["wall_s_total"] = summary["wall_s_total"]


def add_end_to_end_timing(row, manifest, stage1_path, stage2_path, results_dir):
    """Overwrite the row's timing with whole-question (stage 1 + 2) numbers.

    This is what makes the RoG column comparable to ToG's: the same
    aggregate_run_metrics() reads per-question records that cover a whole
    question, so `full_speedup_x` is a full-system speedup in both systems.
    Without both sidecars the row is left with planner_* only -- an absent
    end-to-end number is honest, a planner-only one relabelled as full-system
    is not.
    """
    if not stage1_path or not stage2_path:
        missing = "stage 1" if not stage1_path else "stage 2"
        print(f"  [warn] no end-to-end timing for {manifest['tag']}: "
              f"{missing} per-question metrics missing (re-run to record them)")
        return

    merged = Path(results_dir) / "e2e_metrics" / f"{manifest['tag']}.jsonl"
    timing, summary, _breakdown, dropped = aggregate_end_to_end(
        str(stage1_path), str(stage2_path), merged_path=str(merged)
    )
    if not timing:
        print(f"  [warn] no joined questions for {manifest['tag']}")
        return

    row.update(timing)          # hits/misses/avg_* and full_speedup_x, end-to-end
    row["wall_s_total"] = summary["wall_s_total"]
    row["n_questions"] = summary["n_questions"]
    row["hit_rate"] = summary["hit_rate"]
    row["e2e_metrics_file"] = summary["e2e_metrics_file"]
    if dropped:
        row["questions_dropped_no_stage2"] = len(dropped)
        print(f"  [warn] {manifest['tag']}: {len(dropped)} question(s) timed in "
              f"stage 1 but absent from stage 2; excluded from the end-to-end time")


def build_row(manifest_path, results_dir):
    manifest = json.loads(manifest_path.read_text())
    row = {"tag": manifest["tag"], "policy": manifest["policy"]}

    stats_path = resolve(manifest["cache_stats"], results_dir)
    stats = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        timing = stats.pop("timing", {}) or {}
        # Stage-1 timings go in under planner_* names; the bare names are then
        # free to carry the end-to-end numbers, which is what readers assume
        # `hit_total_s` / `full_speedup_x` already meant.
        for key, value in timing.items():
            if key in LEGACY_STAGE1_TIMING_FIELDS:
                continue
            row[planner_key(key)] = value
        row.update(stats)
        # The planner records policy "off" when caching is disabled; the manifest's
        # name for the same run is "none". The manifest wins.
        row["policy"] = manifest["policy"]
    else:
        print(f"  [warn] missing cache stats: {manifest['cache_stats']}")

    # Both blocks are rebuilt from the per-question sidecars rather than trusted
    # from cache_stats.json, so a resumed policy reports the whole split. Planner
    # first, end-to-end second: where stage 2's sidecar exists, its joined numbers
    # own the bare columns.
    stage1_path, stage2_path = stage_metrics_paths(manifest, stats, results_dir)
    add_planner_timing(row, stage1_path)
    add_end_to_end_timing(row, manifest, stage1_path, stage2_path, results_dir)

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
