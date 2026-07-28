#!/usr/bin/env python3
"""Raw accuracy metrics for the *uncached* baseline (policy 'none'), ToG + RoG.

Collects the policy=='none' row from every run of both systems and prints one
table, in the style of the ToG paper's accuracy table:

    System  Config                          Dataset   EM  Precision  Recall  F1  Hits@1  Correct  Total

Sources (raw, per system -- deliberately not the normalized artifacts/, which
clamp/rescale some fields):

    ToG:  src/ToG-cache/output/compare_results/<tag>/{summary.json, <policy>.jsonl}
          -> summary.json locates each policy's output JSONL; metrics are then
             recomputed from that JSONL with ToG's own eval.py logic, restricted
             to the first pass (loop_idx == 0) so a --loop N>1 run is scored on its
             cold pass alone rather than all passes concatenated. --tog-all-passes
             disables the restriction. Correct = EM-right count, Total = records.
    RoG:  artifacts/rog_cache/<tag>/KGQA/<policy>/.../detailed_eval_result.jsonl
          -> per-question records with prediction / ground_truth / hit. RoG's own
             eval emits NO exact match, so EM here is recomputed by applying ToG's
             exact_match rule (bidirectional substring containment) to RoG's stored
             answers -- the SAME EM definition ToG uses, so the columns are
             comparable. Correct is that EM count, Total the scored-record count.

Corrected precision: both systems inherit RoG's eval_f1, whose precision is
matched/len(prediction) with `matched` counting *gold answers* found in the
prediction -- a gold-count over a prediction-count, so it can exceed 100% (worst
for ToG, which emits one blob prediction: 3 golds in 1 item -> 300%). This report
recomputes precision as (predicted items hitting a gold)/(predicted items), and
F1 from it. Recall, Hits@1 and EM are unaffected. See corrected_prf().

Usage:
    ./scripts/baseline_metrics_table.py
    ./scripts/baseline_metrics_table.py --policy none --csv artifacts/baseline_none.csv
    ./scripts/baseline_metrics_table.py --tog-dir <dir> --rog-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import string
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOG_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "compare_results"
TOG_EVAL_DIR = REPO_ROOT / "src" / "ToG-cache" / "eval"
ROG_DIR = REPO_ROOT / "artifacts" / "rog_cache"

# Columns in report order. 'system'/'config'/'dataset' identify the run; the rest
# are the metrics. Correct/Total are integer counts; the metrics are percentages.
COLUMNS = ["system", "config", "dataset", "em", "precision", "recall", "f1",
           "hits1", "correct", "total"]


def _pct(value):
    """A stored fraction (0-1) as a percentage, or None if absent/non-numeric."""
    return round(value * 100, 2) if isinstance(value, (int, float)) else None


def norm_dataset(name):
    """'RoG-webqsp' / 'webqsp' -> 'WebQSP'; 'cwq' -> 'CWQ'; else best-effort."""
    if not name:
        return "?"
    stem = str(name)
    for prefix in ("RoG-", "rog-", "ToG-", "tog-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    known = {"webqsp": "WebQSP", "cwq": "CWQ", "qald": "QALD",
             "cwq_test": "CWQ", "webqsp_test": "WebQSP"}
    return known.get(stem.lower(), stem)


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] skipping {path}: {exc}")
        return None


def rows_of(doc):
    """Both schemas: a bare list of rows, or {'rows': [...]}."""
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        return doc.get("rows", [])
    return []


def load_tog_eval_utils():
    """Load src/ToG-cache/eval/utils.py by path (stdlib-only: json/re/string).

    Loaded by file to dodge the name clash with ToG/utils.py, mirroring
    summarize_tog_cache.py. Gives us ToG's exact eval functions so first-pass
    numbers match what eval.py would report on the same records.
    """
    spec = importlib.util.spec_from_file_location("bmt_tog_eval_utils", TOG_EVAL_DIR / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tog_metrics_from_jsonl(path, eu, first_pass_only=True):
    """Recompute ToG's eval metrics over a per-config output JSONL.

    Replicates eval.py's per-record loop exactly (EM via exact_match on the raw
    'results' string with the same check_string/clean_results/NULL handling;
    Hits@1/F1/Precision/Recall via the RoG-style helpers). With first_pass_only,
    keep only loop_idx == 0 -- so a --loop N>1 run is scored on its cold first
    pass alone, not all passes concatenated. Records without loop_idx (single-pass
    runs) are all treated as the first pass.
    """
    records = [json.loads(l) for l in Path(path).open() if l.strip()]
    if first_pass_only and any("loop_idx" in r for r in records):
        records = [r for r in records if r.get("loop_idx", 0) == 0]
    if not records:
        return None

    num_right = 0
    hit = f1s = precs = recs = 0.0
    for d in records:
        answers = d.get("ground_truth") or []
        results = d.get("results")
        prediction = eu.prediction_to_list(results)
        hit += eu.rog_eval_hit(prediction, answers)
        f1, prec, rec = corrected_prf(prediction, answers)  # corrected precision
        f1s += f1
        precs += prec
        recs += rec
        # Exact Match, byte-for-byte with eval.py: when check_string(results) is
        # true and the cleaned result is "NULL", the record scores neither right
        # nor wrong (it still counts toward n); constraints_refuse's skip never
        # fires here because it's guarded by check_string in the else branch.
        if eu.check_string(results):
            response = eu.clean_results(results)
            if response != "NULL" and eu.exact_match(response, answers):
                num_right += 1
        elif eu.exact_match(results, answers):
            num_right += 1

    n = len(records)
    return {
        "em": round(100 * num_right / n, 2),
        "precision": round(100 * precs / n, 2),
        "recall": round(100 * recs / n, 2),
        "f1": round(100 * f1s / n, 2),
        "hits1": round(100 * hit / n, 2),
        "correct": num_right,
        "total": n,
    }


def collect_tog(tog_dir, policy, first_pass_only=True):
    eu = load_tog_eval_utils()
    out = []
    for summary in sorted(Path(tog_dir).glob("*/summary.json")):
        doc = load_json(summary)
        if doc is None:
            continue
        dataset = (doc.get("args", {}) or {}).get("dataset") if isinstance(doc, dict) else None
        for r in rows_of(doc):
            if r.get("policy") != policy:
                continue
            row = {"system": "ToG", "config": summary.parent.name,
                   "dataset": norm_dataset(dataset)}
            out_path = r.get("output")
            metrics = tog_metrics_from_jsonl(out_path, eu, first_pass_only) if out_path \
                and Path(out_path).exists() else None
            if metrics is None:
                # Fall back to the pre-aggregated summary row (all passes) so the
                # config still appears; flag it so the caller knows it's not
                # first-pass-restricted.
                print(f"  [warn] {summary.parent.name}: no readable output JSONL "
                      f"({out_path}); using all-pass summary values")
                metrics = {
                    "em": _pct(r.get("exact_match")), "precision": _pct(r.get("precision")),
                    "recall": _pct(r.get("recall")), "f1": _pct(r.get("f1")),
                    "hits1": _pct(r.get("hits1")), "correct": r.get("right"),
                    "total": r.get("records"),
                }
            out.append({**row, **metrics})
    return out


def rog_normalize(s):
    """Mirror of eval/utils.py rog_normalize: lowercase, strip punctuation and
    the articles a/an/the, collapse whitespace."""
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def rog_match(haystack, needle):
    """True if `needle` occurs in `haystack` after normalization (utils.py rog_match)."""
    return rog_normalize(needle) in rog_normalize(haystack)


def corrected_prf(prediction, answer):
    """(f1, precision, recall) with a CORRECT precision.

    Both ToG and RoG inherit RoG's eval_f1, whose precision is matched/len(pred)
    where `matched` counts *gold answers* found in the joined prediction -- a
    gold-count over a prediction-count, so it can exceed 1.0 (e.g. one predicted
    blob containing 3 golds -> 3/1 = 300%). Fixed here:

        precision = (# predicted items that hit some gold) / (# predicted items)
        recall    = (# gold answers found in the prediction) / (# gold answers)

    Recall is unchanged from RoG's canonical formula; only precision (and the F1
    derived from it) is corrected, so precision is now bounded by 1.0.
    """
    if not prediction or not answer:
        return 0.0, 0.0, 0.0
    # Normalize each string once (predictions can be long -> O(pred*gold)
    # re-normalization is what made this slow).
    npred = [rog_normalize(p) for p in prediction]
    njoined = rog_normalize(" ".join(prediction))
    ngold = [rog_normalize(a) for a in answer]
    recall_hits = sum(1 for a in ngold if a in njoined)
    precision_hits = sum(1 for p in npred if any(a in p for a in ngold))
    precision = precision_hits / len(prediction)
    recall = recall_hits / len(answer)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def tog_exact_match(response, answers):
    """Mirror of src/ToG-cache/eval/utils.py exact_match (bidirectional substring
    containment on space-stripped, lowercased strings). Kept inline so this script
    stays dependency- and import-path-free."""
    clean = response.strip().replace(" ", "").lower()
    for answer in answers:
        ca = answer.strip().replace(" ", "").lower()
        if clean == ca or clean in ca or ca in clean:
            return True
    return False


def _mean_pct(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(100 * sum(nums) / len(nums), 2) if nums else None


def metrics_from_detailed(path):
    """Compute the report columns from one RoG detailed_eval_result.jsonl.

    EM/Correct use ToG's exact_match rule over the stored answer list (RoG's eval
    does not emit EM); Precision/Recall/F1/Hits@1 average the per-record fields.
    RoG's data misspells the precision key as 'precission'; accept either.
    """
    records = [json.loads(l) for l in path.open() if l.strip()]
    if not records:
        return None
    correct = 0
    f1s = precs = recs = 0.0
    for r in records:
        pred = r.get("prediction") if isinstance(r.get("prediction"), list) else []
        gold = r.get("ground_truth") or []
        if pred and gold and any(tog_exact_match(p, gold) for p in pred):
            correct += 1
        # Recompute P/R/F1 with corrected precision rather than trusting the
        # stored 'precission' field, which uses RoG's buggy formula.
        f1, prec, rec = corrected_prf(pred, gold)
        f1s += f1
        precs += prec
        recs += rec
    n = len(records)
    return {
        "em": round(100 * correct / n, 2),
        "precision": round(100 * precs / n, 2),
        "recall": round(100 * recs / n, 2),
        "f1": round(100 * f1s / n, 2),
        "hits1": _mean_pct([r.get("hit") for r in records]),
        "correct": correct,
        "total": n,
    }


def collect_rog(rog_dir, policy):
    """One row per (tag, dataset): metrics computed from that run's uncached
    detailed_eval_result.jsonl. Path shape:
        <rog_dir>/<tag>/KGQA/<policy_folder>/<Dataset>/RoG/<split>/.../detailed_eval_result.jsonl
    <policy_folder> is the bare policy for 'none', or '<policy>_t<thr>' for the
    semantic ones -- matched by prefix. When a tag has several splits (e.g. a
    'test[:10]' smoke run beside the full 'test'), keep the largest.
    """
    root = Path(rog_dir)
    if not root.is_dir():
        return []
    best = {}
    for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if tag_dir.name in ("KGQA", "gen_rule_path"):
            continue  # stray top-level eval trees, not run tags
        for det in tag_dir.rglob("detailed_eval_result.jsonl"):
            parts = det.parts
            if "KGQA" not in parts:
                continue
            i = parts.index("KGQA")
            if i + 2 >= len(parts) or not parts[i + 1].startswith(policy):
                continue
            metrics = metrics_from_detailed(det)
            if metrics is None:
                continue
            dataset = norm_dataset(parts[i + 2])
            key = (tag_dir.name, dataset)
            if key in best and best[key]["total"] >= metrics["total"]:
                continue  # keep the fullest split for this tag/dataset
            best[key] = {"system": "RoG", "config": tag_dir.name,
                         "dataset": dataset, **metrics}
    return list(best.values())


def fmt(value, width, is_count=False):
    if value is None:
        return f"{'-':>{width}}"
    if is_count:
        return f"{int(value):>{width}}"
    return f"{value:>{width}.2f}"


def print_table(rows):
    if not rows:
        print("no matching rows found.")
        return
    sys_w = max(6, max(len(r["system"]) for r in rows))
    cfg_w = max(6, max(len(r["config"]) for r in rows))
    ds_w = max(7, max(len(r["dataset"]) for r in rows))
    header = (f"{'System':<{sys_w}}  {'Config':<{cfg_w}}  {'Dataset':<{ds_w}}  "
              f"{'EM':>8}  {'Precision':>9}  {'Recall':>8}  {'F1':>8}  "
              f"{'Hits@1':>8}  {'Correct':>8}  {'Total':>8}")
    print(header)
    print("-" * len(header))
    # ToG first, then RoG; within a system by dataset then config.
    for r in sorted(rows, key=lambda x: (x["system"] != "ToG", x["dataset"], x["config"])):
        print(f"{r['system']:<{sys_w}}  {r['config']:<{cfg_w}}  {r['dataset']:<{ds_w}}  "
              f"{fmt(r['em'], 8)}  {fmt(r['precision'], 9)}  {fmt(r['recall'], 8)}  "
              f"{fmt(r['f1'], 8)}  {fmt(r['hits1'], 8)}  "
              f"{fmt(r['correct'], 8, is_count=True)}  {fmt(r['total'], 8, is_count=True)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="none",
                    help="cache policy to report (default: none = uncached baseline)")
    ap.add_argument("--tog-dir", default=str(TOG_DIR),
                    help="dir of ToG compare_results/<tag>/summary.json")
    ap.add_argument("--rog-dir", default=str(ROG_DIR),
                    help="dir of RoG run tags (each holds KGQA/.../detailed_eval_result.jsonl)")
    ap.add_argument("--tog-all-passes", action="store_true",
                    help="score ToG over all loop passes (default: only the first "
                         "pass, loop_idx == 0, so looped runs aren't double-counted)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the table to this CSV path")
    args = ap.parse_args()

    first_pass = not args.tog_all_passes
    rows = (collect_tog(args.tog_dir, args.policy, first_pass_only=first_pass)
            + collect_rog(args.rog_dir, args.policy))

    scope = "all passes" if args.tog_all_passes else "first pass only"
    print(f"\nRaw baseline metrics  (policy = {args.policy!r}, ToG: {scope})")
    print(f"  ToG source: {args.tog_dir}")
    print(f"  RoG source: {args.rog_dir}\n")
    print_table(rows)
    print(f"\nNote: Precision is CORRECTED here -- both systems' eval divides matched "
          "golds by the\nnumber of predicted items, which can exceed 100% (see "
          "corrected_prf); this report uses\n(predicted items hitting a gold)/(predicted "
          "items) and recomputes F1 from it. Recall,\nHits@1 and EM are unchanged. ToG "
          f"metrics come from each run's output JSONL ({scope}) via\nToG's own eval.py "
          "logic; RoG's EM/Correct apply ToG's exact_match rule to RoG's stored\nanswers "
          "(same EM definition, so the columns are comparable).")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for r in sorted(rows, key=lambda x: (x["system"] != "ToG", x["dataset"], x["config"])):
                writer.writerow(r)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
