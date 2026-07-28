#!/usr/bin/env python3
"""Emit a ToG cache-experiment summary in the RoG plot schema.

The ToG twin of src/RoG-cache/summarize_rog_cache.py. plot_rog_cache_results.py
plots one line per run over the cache-policy axis, reading each run's
summary.json (a list of per-policy rows with policy, hit, f1, accuracy, hit_rate,
speedup_x, full_speedup_x, ...). ToG's own experiment
(compare_cache_accuracy.py) instead produces baseline-vs-cache *configs* for a
single policy, so this joins its two sources into that per-policy schema:

    compare_results/<tag>/summary.json   ->  accuracy (Hits@1 / F1 / recall)
    <config>.jsonl.metrics.jsonl         ->  hit rate, speedup, LLM calls
                                             (restart-safe; see ToG/cache_metrics.py)

compare_cache_accuracy.py already sweeps one config per cache policy (none / exact
/ semantic_lfu / semantic_lru / semantic_oracle), so each row maps straight to a
policy point and a ToG run shows up as a full line alongside the RoG lines.
Accuracy is rescaled 0-1 -> 0-100 to match RoG's percentage convention.

    # one run tag under src/ToG-cache/output/compare_results/
    ./scripts/summarize_tog_cache.py --run gemini_tog_cache_oxi_test

    # then overlay on the RoG figure
    ./scripts/plot_rog_cache_results.py \
        --runs gemini_rog_cache_oxi_test --tog-runs gemini_tog_cache_oxi_test

With --per-pass (only meaningful for a --loop N>1 run), one summary per pass is
written to artifacts/tog_cache/<tag>_pass<K>/, so pass 1 (cold cache) and pass 2
(warm) can be plotted as separate lines:

    ./scripts/summarize_tog_cache.py --run <tag> --per-pass
    ./scripts/plot_rog_cache_results.py --tog-runs <tag>_pass1 <tag>_pass2

Writes artifacts/tog_cache/<tag>/summary.json (same layout as artifacts/rog_cache).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOG_ROOT = REPO_ROOT / "src" / "ToG-cache" / "ToG"
EVAL_ROOT = REPO_ROOT / "src" / "ToG-cache" / "eval"
COMPARE_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "compare_results"
OUT_DIR = REPO_ROOT / "artifacts" / "tog_cache"

# cache_metrics is stdlib-only, so importing it here pulls in no ML deps.
sys.path.insert(0, str(TOG_ROOT))
from cache_metrics import aggregate_run_metrics, metrics_sidecar_path  # noqa: E402


def load_eval_utils():
    """Load eval/utils.py (the RoG-compatible metric fns) by path.

    Loaded by file to dodge the name clash with ToG/utils.py; it is stdlib-only.
    """
    spec = importlib.util.spec_from_file_location("tog_eval_utils", EVAL_ROOT / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def accuracy_from_records(records, eu):
    """RoG-schema accuracy (%) computed from answer records with inline ground_truth."""
    hits, f1s, precs, recs = [], [], [], []
    for r in records:
        gold = r.get("ground_truth")
        if not gold:
            continue  # need inline gold to score this pass's records
        pred = eu.prediction_to_list(r.get("results"))
        hits.append(eu.rog_eval_hit(pred, gold))
        f1, p, rc = eu.rog_eval_f1(pred, gold)
        f1s.append(f1)
        precs.append(p)
        recs.append(rc)
    n = len(hits) or 1
    return {
        "hit": round(100 * sum(hits) / n, 2),
        "f1": round(100 * sum(f1s) / n, 2),
        "precision": round(100 * sum(precs) / n, 2),
        "recall": round(100 * sum(recs) / n, 2),
        "accuracy": round(100 * sum(recs) / n, 2),  # RoG's "accuracy" ~ recall
        "records": len(hits),
    }


def cache_from_per_loop(pl, base_miss_s):
    """RoG-schema cache/timing fields for a single pass (one per_loop entry).

    `base_miss_s` is the run's overall average miss (cold) time -- the cost of a
    question with no cache. Using it as the no-cache reference makes a warm pass's
    speedup well defined even when that pass has zero misses (all hits), which is
    exactly the pass where speedup is largest.
    """
    hits, misses = pl["hits"], pl["misses"]
    hit_s, miss_s = pl["hit_total_s"], pl["miss_total_s"]
    n = hits + misses
    actual_s = hit_s + miss_s
    avg_hit = (hit_s / hits) if hits else 0.0
    avg_miss = (miss_s / misses) if misses else 0.0
    # speedup vs a no-cache run where every question cost base_miss_s.
    speedup = (base_miss_s / avg_hit) if (hits and avg_hit > 0 and base_miss_s > 0) else None
    full_speedup = ((n * base_miss_s) / actual_s
                    if (n and actual_s > 0 and base_miss_s > 0) else None)
    return {
        "hits": hits, "misses": misses,
        "hit_total_s": round(hit_s, 3), "miss_total_s": round(miss_s, 3),
        "avg_hit_s": round(avg_hit, 3), "avg_miss_s": round(avg_miss, 3),
        "speedup_x": round(speedup, 2) if speedup is not None else None,
        "full_speedup_x": round(full_speedup, 2) if full_speedup is not None else None,
        "n_questions": n,
        "hit_rate": (hits / n) if n else 0.0,
    }


def passes_present(comp_rows):
    """Sorted set of loop indices found across the configs' metrics sidecars."""
    passes = set()
    for cfg in comp_rows:
        _t, _s, _b, per_loop = aggregate_run_metrics(metrics_sidecar_path(cfg.get("output")))
        passes.update(pl["loop"] for pl in per_loop)
    return sorted(passes)


def build_pass_row(cfg, policy, tag, pass_idx, eu):
    """RoG-schema row for one policy on one pass."""
    out = cfg.get("output")
    overall, _s, _b, per_loop = aggregate_run_metrics(metrics_sidecar_path(out) if out else None)
    # Cold-cost reference: the run's overall avg miss time (pass 1 dominates misses).
    base_miss_s = overall.get("avg_miss_s") or 0.0
    pl = next((x for x in per_loop if x["loop"] == pass_idx), None)
    row = {"policy": policy, "tag": f"{tag}_pass{pass_idx + 1}",
           "config": cfg.get("config"), "loop": pass_idx}
    row.update(cache_from_per_loop(pl, base_miss_s) if pl
               else {"hit_rate": 0.0, "speedup_x": None, "full_speedup_x": None, "n_questions": 0})
    row.update(accuracy_from_records(
        [r for r in read_jsonl(out) if r.get("loop_idx") == pass_idx], eu))
    return row


def resolve_compare_dir(run: str) -> Path:
    """Accept a tag, a dir, or a path to compare_results/<tag>/summary.json."""
    p = Path(run)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    return COMPARE_DIR / run


def build_row(config_row: dict, policy: str, tag: str) -> dict:
    """Join one compare config row + its metrics sidecar into a RoG-schema row."""
    # Accuracy: compare stores fractions (0-1); RoG stores percentages (0-100).
    def pct(key):
        v = config_row.get(key)
        return round(v * 100, 2) if isinstance(v, (int, float)) else None

    row = {"policy": policy, "tag": tag, "config": config_row.get("config")}

    out_path = config_row.get("output")
    metrics_path = metrics_sidecar_path(out_path) if out_path else None
    timing, summary, _breakdown, _per_loop = aggregate_run_metrics(metrics_path)
    if summary["n_questions"] == 0:
        print(f"  [warn] no metrics sidecar for config {config_row.get('config')!r} "
              f"({metrics_path}); hit_rate/speedup will be blank")

    row.update(timing)              # hits/misses/avg_*/speedup_x/full_speedup_x
    row.update(summary)             # n_questions/hit_rate/llm_calls*/wall_s_total
    row.update({
        "hit": pct("hits1"),        # Hits@1 (%)
        "f1": pct("f1"),            # F1 (%)
        "accuracy": pct("recall"),  # RoG's "accuracy" ~ recall of gold answers
        "precision": pct("precision"),
        "recall": pct("recall"),
        "exact_match": config_row.get("exact_match"),
        "records": config_row.get("records"),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="compare_results tag, dir, or summary.json path")
    ap.add_argument("--tag", default=None,
                    help="output tag (default: the run dir name)")
    ap.add_argument("--per-pass", action="store_true",
                    help="for a --loop N>1 run, write one summary per pass to "
                         "artifacts/tog_cache/<tag>_pass<K>/ (cold pass 1 vs warm "
                         "pass 2), each plottable as its own line.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output summary.json (combined mode only; default: "
                         "artifacts/tog_cache/<tag>/summary.json)")
    args = ap.parse_args()

    compare_dir = resolve_compare_dir(args.run)
    summary_path = compare_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"no compare summary at {summary_path}; run the ToG cache "
                         f"experiment first (scripts/run_tog_cache_experiment.py)")

    compare = json.loads(summary_path.read_text())
    comp_rows = compare.get("rows", [])
    if not any("hits1" in r for r in comp_rows):
        raise SystemExit(
            f"{summary_path} has no Hits@1/F1 fields -- it predates the metric "
            f"changes. Re-run the ToG cache experiment to regenerate it.")

    tag = args.tag or (compare_dir.name if compare_dir.name != "compare_results" else "tog")

    if args.per_pass:
        passes = passes_present(comp_rows)
        if len(passes) <= 1:
            raise SystemExit(
                "only one pass found (loop_idx). --per-pass needs a --loop N>1 run "
                "produced by the updated code; nothing to split.")
        eu = load_eval_utils()
        written = []
        for p in passes:
            rows = [build_pass_row(cfg, cfg.get("policy") or cfg.get("config"), tag, p, eu)
                    for cfg in comp_rows]
            rows.sort(key=lambda r: r["policy"] != "none")
            pass_tag = f"{tag}_pass{p + 1}"
            out_path = OUT_DIR / pass_tag / "summary.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=2))
            written.append(pass_tag)
            print(f"wrote {out_path}  ({len(rows)} policy rows)")
        print("\nplot both passes with:\n  ./scripts/plot_rog_cache_results.py "
              f"--tog-runs {' '.join(written)}")
        return

    # Combined mode: one RoG-schema row per swept policy (config name == policy).
    out_rows = [build_row(cfg, cfg.get("policy") or cfg.get("config"), tag)
                for cfg in comp_rows]
    if not out_rows:
        raise SystemExit(f"no policy rows found in {summary_path}")

    # baseline ('none') first, matching RoG's summary ordering.
    out_rows.sort(key=lambda r: r["policy"] != "none")

    out_path = args.out or (OUT_DIR / tag / "summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, indent=2))
    print(f"wrote {out_path}  ({len(out_rows)} policy rows: "
          f"{', '.join(r['policy'] for r in out_rows)})")


if __name__ == "__main__":
    main()
