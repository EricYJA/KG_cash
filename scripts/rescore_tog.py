#!/usr/bin/env python3
"""Rescore a ToG cache run's answers without touching eval/utils.py.

Two problems this works around, both in `clean_results` (src/ToG-cache/eval/utils.py):

  1. It scans to the FIRST `{`...`}`. ToG's `prompt_evaluate` few-shots
     (ToG/prompt_list.py:66) teach the shape
         A: {Yes}. ... the answer to the question is {De Smet}.
     and main_freebase.py:270 saves that whole response as `results`, so the
     extracted "answer" is the sufficiency marker `Yes`. In a live WebQSP run
     that hit 520 of 1639 records and cost ~25 points of Hits@1.
  2. An empty `{}` yields "", and `exact_match` does `clean_result in
     clean_answer` -- "" is a substring of every gold answer, so those records
     were counted correct unconditionally.

Fixing them in place would change scoring under a run that is still going:
compare_cache_accuracy.py:239 scores each policy as its loop reaches it, so a
mid-run patch leaves some rows on the old extractor and some on the new one.
This script instead imports only the *pure* helpers from eval/utils.py
(rog_eval_hit / rog_eval_f1 / exact_match) and applies its own extraction, so
nothing a running job reads is modified.

It scores every policy carrying a `<policy>.done` marker, so it works on a run
in flight (4 of 5 policies) and again unchanged once the run completes (5 of 5).

    ./scripts/rescore_tog.py --run tog_rerun_live_virt
    ./scripts/rescore_tog.py --all
    ./scripts/rescore_tog.py --all --scorer stock   # reproduce the buggy numbers

Writes compare_results/<tag>_rescored/summary.json in compare_cache_accuracy's
schema, so it feeds the normal downstream chain unchanged:

    ./scripts/summarize_tog_cache.py --run <that dir> --tag <tag>
    ./scripts/plot_rog_cache_results.py --tog-runs <tag>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "compare_results"
EVAL_UTILS = REPO_ROOT / "src" / "ToG-cache" / "eval" / "utils.py"

BRACES = re.compile(r"\{([^{}]*)\}")


def load_eval_utils():
    """ToG's own metric functions, so numbers stay comparable to eval.py."""
    spec = importlib.util.spec_from_file_location("tog_eval_utils", EVAL_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def answer_fixed(results: str) -> str | None:
    """Last braced span that is not the Yes/No marker and not empty."""
    spans = [s.strip() for s in BRACES.findall(results)]
    spans = [s for s in spans if s and s.lower() not in ("yes", "no")]
    return spans[-1] if spans else None


def answer_stock(results: str) -> str | None:
    """What eval.py's clean_results extracts today (first span, may be empty)."""
    if "{" not in results:
        return None
    start, end = results.find("{") + 1, results.find("}")
    return results[start:end]


def score(records: list[dict], eu, scorer: str) -> dict:
    """Per-run metrics in eval.py's output schema.

    Unlike eval.py every record is scored: its `response == "NULL"` branch falls
    through without counting, which is unreachable under the stock extractor but
    would silently drop records under the fixed one.
    """
    pick = answer_stock if scorer == "stock" else answer_fixed
    right = error = 0
    hits: list[int] = []
    f1s: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    no_answer = 0

    for d in records:
        gold = d.get("ground_truth") or []
        results = d.get("results") or ""
        span = pick(results)
        if span is None or not span.strip():
            no_answer += 1
            span = results          # fall back to the prose, as eval.py does
        prediction = [p.strip() for p in span.split("\n") if p.strip()] or [results]

        hits.append(eu.rog_eval_hit(prediction, gold))
        f1, prec, rec = eu.rog_eval_f1(prediction, gold)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        if eu.exact_match(span, gold):
            right += 1
        else:
            error += 1

    n = len(records) or 1
    return {"exact_match": right / n, "right": right, "error": error,
            "records": len(records), "hits1": sum(hits) / n, "f1": sum(f1s) / n,
            "precision": sum(precs) / n, "recall": sum(recs) / n,
            "no_answer": no_answer}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def rescore_run(src: Path, eu, scorer: str) -> list[dict]:
    """One row per completed policy, in compare_cache_accuracy's row schema."""
    rows = []
    for marker in sorted(src.glob("*.done")):
        policy = marker.stem
        jsonl = src / f"{policy}.jsonl"
        if not jsonl.exists():
            print(f"  [skip] {policy}: no {jsonl.name}")
            continue
        records = read_jsonl(jsonl)
        missing = sum(1 for d in records if not d.get("ground_truth"))
        if missing:
            print(f"  [note] {policy}: {missing} record(s) with empty ground_truth "
                  f"(counted in the denominator, never a hit)")
        stock = score(records, eu, "stock")
        fixed = score(records, eu, scorer)
        print(f"  {policy:<16} n={fixed['records']:>5}  "
              f"Hits@1 {stock['hits1']:.4f} -> {fixed['hits1']:.4f}   "
              f"EM {stock['exact_match']:.4f} -> {fixed['exact_match']:.4f}   "
              f"F1 {stock['f1']:.4f} -> {fixed['f1']:.4f}")
        rows.append({"config": policy, "policy": policy, **fixed,
                     "output": str(jsonl)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="compare_results tag or directory")
    g.add_argument("--all", action="store_true",
                   help="every compare_results/<tag>/ holding a *.done marker "
                        "(skips this script's own _rescored staging dirs)")
    ap.add_argument("--scorer", default="fixed", choices=["fixed", "stock"],
                    help="fixed (default): last non-Yes/No braced span. "
                         "stock: reproduce eval.py's first-span behaviour.")
    ap.add_argument("--suffix", default="_rescored",
                    help="staging dir suffix beside the run (default: _rescored)")
    args = ap.parse_args()

    if args.all:
        runs = sorted(d for d in COMPARE_DIR.iterdir()
                      if d.is_dir() and not d.name.endswith(args.suffix)
                      and any(d.glob("*.done")))
    else:
        p = Path(args.run)
        runs = [p if p.is_dir() else COMPARE_DIR / args.run]

    eu = load_eval_utils()
    for src in runs:
        if not src.is_dir():
            raise SystemExit(f"no such run dir: {src}")
        print(f"\n=== {src.name}")
        rows = rescore_run(src, eu, args.scorer)
        if not rows:
            print("  (no completed policies)")
            continue
        out_dir = src.parent / f"{src.name}{args.suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        identity = {}
        cfg = src / "run_config.json"
        if cfg.exists():
            identity = json.loads(cfg.read_text()).get("identity", {})
        (out_dir / "summary.json").write_text(json.dumps(
            {"args": {"dataset": identity.get("dataset", "webqsp"),
                      "scorer": args.scorer, "rescored_from": str(src),
                      "identity": identity},
             "rows": rows}, indent=2))
        print(f"  wrote {out_dir / 'summary.json'}  "
              f"({len(rows)} policies: {', '.join(r['policy'] for r in rows)})")
        print(f"  next: ./scripts/summarize_tog_cache.py --run {out_dir} "
              f"--tag {src.name}")


if __name__ == "__main__":
    main()
