#!/usr/bin/env python3
"""Compare a normal (t=0.90) live run against the mislabelled `_wrong` (t=0.50) run.

One row per system (ToG, RoG) for a single semantic cache policy, in two tables:
accuracy and Hits@1. The normal run is the left column; the `_wrong` run is
scored against it.

Both sides are rescored from their raw prediction files over the question ids
the two runs SHARE -- the wrong runs are `test[:400]` while the normal runs are
the full split, and neither file is complete, so the published summaries sit on
different denominators and cannot be subtracted.

    ./scripts/compare_wrong_threshold.py --policy semantic_lfu

Only threshold-bearing policies are accepted -- `none` and `exact` do not read
the similarity threshold, so a threshold comparison is meaningless for them.

Metrics keep each system's own definition, so the numbers stay comparable to
that system's published tables: ToG accuracy is eval/utils.py `exact_match` on
the extracted span, RoG accuracy is RoG's `eval_acc` (matched gold / gold).
Hits@1 is the same `rog_eval_hit` on both sides.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROG_DIR = REPO_ROOT / "artifacts" / "rog_cache"
TOG_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "compare_results"
EVAL_UTILS = REPO_ROOT / "src" / "ToG-cache" / "eval" / "utils.py"

BRACES = re.compile(r"\{([^{}]*)\}")
THRESHOLDLESS = {"none", "exact"}
WIDTH = 70


def load_eval_utils():
    """ToG's own metric helpers; its rog_* functions mirror RoG's evaluator."""
    spec = importlib.util.spec_from_file_location("tog_eval_utils", EVAL_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def tog_span(results: str, scorer: str) -> str:
    """The answer ToG's evaluator reads out of a response (see rescore_tog.py)."""
    if scorer == "stock":
        if "{" not in results:
            return results
        span = results[results.find("{") + 1:results.find("}")]
        return span if span.strip() else results
    spans = [s.strip() for s in BRACES.findall(results)]
    spans = [s for s in spans if s and s.lower() not in ("yes", "no")]
    return spans[-1] if spans else results


def load_tog(run: str, policy: str, scorer: str) -> tuple[dict[str, dict], dict]:
    path = TOG_DIR / run / f"{policy}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path}: missing (no {policy!r} in run {run!r}?)")
    by_id = {}
    for d in read_jsonl(path):
        span = tog_span(d.get("results") or "", scorer)
        pred = [p.strip() for p in span.split("\n") if p.strip()] or [span]
        # A repeated id means the run resumed and re-answered; keep the last.
        by_id[d["id"]] = {"gold": d.get("ground_truth") or [], "pred": pred,
                          "span": span}
    cfg = json.loads((TOG_DIR / run / "run_config.json").read_text())
    ident = cfg.get("identity", cfg)
    return by_id, {"threshold": float(ident["similarity_threshold"]),
                   "model": ident.get("model", "")}


def load_rog(run: str, policy: str) -> tuple[dict[str, dict], dict]:
    meta = json.loads((ROG_DIR / run / "summary.json").read_text())
    meta = [m for m in meta if m["policy"] == policy]
    if not meta:
        raise SystemExit(f"{ROG_DIR / run}: no {policy!r} in summary.json")
    meta = meta[-1]
    tag = meta["tag"]
    hits = sorted((ROG_DIR / run / "KGQA" / tag).glob("**/predictions.jsonl"))
    if not hits:
        raise SystemExit(f"{ROG_DIR / run / 'KGQA' / tag}: no predictions.jsonl")
    by_id = {}
    for d in read_jsonl(hits[0]):
        pred = d.get("prediction") or ""
        if not isinstance(pred, list):
            pred = pred.split("\n")
        by_id[d["id"]] = {"gold": d.get("ground_truth") or [],
                          "pred": [p.strip() for p in pred if p.strip()]}
    return by_id, {"threshold": float(meta["similarity_threshold"]),
                   "model": meta.get("model", "")}


def score(by_id: dict, common: list[str], system: str, eu) -> dict:
    """Accuracy and Hits@1 over `common` only, in the system's own definition."""
    acc = hit = 0.0
    for qid in common:
        rec = by_id[qid]
        gold = rec["gold"]
        hit += eu.rog_eval_hit(rec["pred"], gold)
        if system == "ToG":
            acc += 1.0 if eu.exact_match(rec["span"], gold) else 0.0
        else:
            # RoG's eval_acc: fraction of gold answers found in the prediction.
            joined = " ".join(rec["pred"])
            acc += (sum(1 for a in gold if eu.rog_match(joined, a)) / len(gold)
                    if gold else 0.0)
    n = len(common) or 1
    return {"accuracy": 100 * acc / n, "hits1": 100 * hit / n}


def table(title: str, key: str, pairs: list) -> None:
    norm_h = "t=" + "/".join(sorted({f"{p['normal_t']:.2f}" for p in pairs}))
    wrong_h = "t=" + "/".join(sorted({f"{p['wrong_t']:.2f}" for p in pairs}))
    print("\n" + "=" * WIDTH)
    print(title)
    print(f"{'system':<10}{norm_h + ' (normal)':>21}{wrong_h + ' (wrong)':>21}"
          f"{'delta':>10}")
    print("-" * WIDTH)
    for p in pairs:
        print(f"{p['system']:<10}{p['normal'][key]:>21.2f}{p['wrong'][key]:>21.2f}"
              f"{p['wrong'][key] - p['normal'][key]:>+10.2f}")
    print("=" * WIDTH)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True,
                    help="cache policy to compare, e.g. semantic_lfu")
    ap.add_argument("--rog-normal", default="rog_live_virt_gemini")
    ap.add_argument("--rog-wrong", default="rog_wrong")
    ap.add_argument("--tog-normal", default="tog_rerun_live_virt_gemini")
    ap.add_argument("--tog-wrong", default="tog_wrong")
    ap.add_argument("--scorer", default="fixed", choices=["fixed", "stock"],
                    help="ToG answer extractor; 'stock' reproduces the buggy numbers")
    ap.add_argument("--csv", type=Path, help="also write both tables as CSV")
    args = ap.parse_args()

    if args.policy in THRESHOLDLESS:
        raise SystemExit(
            f"--policy {args.policy}: the similarity threshold does not apply to "
            f"{'/'.join(sorted(THRESHOLDLESS))}, so there is nothing to compare"
        )

    eu = load_eval_utils()
    sources = [
        ("ToG", args.tog_normal, args.tog_wrong,
         lambda run: load_tog(run, args.policy, args.scorer)),
        ("RoG", args.rog_normal, args.rog_wrong,
         lambda run: load_rog(run, args.policy)),
    ]

    print(f"\npolicy: {args.policy}   (ToG extractor: {args.scorer})")
    pairs = []
    for system, normal_run, wrong_run, load in sources:
        normal, n_meta = load(normal_run)
        wrong, w_meta = load(wrong_run)
        common = sorted(set(normal) & set(wrong))
        if not common:
            raise SystemExit(f"{system}: {normal_run} and {wrong_run} share no ids")
        if n_meta["model"] != w_meta["model"]:
            print(f"  [WARN] {system}: model mismatch -- normal {n_meta['model']!r} "
                  f"vs wrong {w_meta['model']!r}")
        pairs.append({
            "system": system,
            "normal_run": normal_run, "wrong_run": wrong_run,
            "normal_t": n_meta["threshold"], "wrong_t": w_meta["threshold"],
            "n_common": len(common),
            "normal": score(normal, common, system, eu),
            "wrong": score(wrong, common, system, eu),
        })
        print(f"  {system:<4} {normal_run} (t={n_meta['threshold']:g}, "
              f"{len(normal)} answered)  vs  {wrong_run} "
              f"(t={w_meta['threshold']:g}, {len(wrong)} answered)")
        print(f"       scored on {len(common)} shared questions; "
              f"model {n_meta['model']}")

    table("Accuracy (%)", "accuracy", pairs)
    table("Hits@1 (%)", "hits1", pairs)
    print("Percentages over the questions the two runs share; "
          "delta is wrong - normal.")
    print("ToG accuracy is exact_match, RoG accuracy is eval_acc "
          "(matched gold / gold).")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "system", "policy", "n_common",
                        "normal_run", "normal_threshold", "normal_value",
                        "wrong_run", "wrong_threshold", "wrong_value", "delta"])
            for key in ("accuracy", "hits1"):
                for p in pairs:
                    w.writerow([key, p["system"], args.policy, p["n_common"],
                                p["normal_run"], p["normal_t"],
                                f"{p['normal'][key]:.2f}", p["wrong_run"],
                                p["wrong_t"], f"{p['wrong'][key]:.2f}",
                                f"{p['wrong'][key] - p['normal'][key]:+.2f}"])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
