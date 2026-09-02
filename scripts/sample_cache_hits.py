#!/usr/bin/env python3
"""Pull example semantic cache hits out of a run, by cosine similarity band.

The point of a low threshold run (`rog_wrong`, t=0.5) is that it accepts hits
the normal t=0.90 runs would reject. The summary tables say how MANY and what
they cost in accuracy; this prints the hits themselves, so the two ends of the
band can be read side by side:

  low  (sim < 0.60)  -- barely-related questions whose relation-path schema was
                        transplanted anyway; this is where a threshold that low
                        does its damage
  high (sim >= 0.90) -- near-paraphrases, the hits a normal run would also take

For each hit: the asked question, the cached question it matched, the cosine
similarity, the relation paths that were served instead of planned, and -- when
the stage-2 answers are on disk -- the final answer, the gold answers, and
whether it was a Hits@1.

    ./scripts/sample_cache_hits.py                        # rog_wrong, 3 + 3
    ./scripts/sample_cache_hits.py -n 5 --sample random
    ./scripts/sample_cache_hits.py --run rog_live_virt_gemini --high-min 0.95

Hits come from stage 1 (`gen_rule_path/<tag>/**/predictions_*.jsonl`), which
records `cache.similarity` and `cache.source_question` per question; answers
come from the matching `KGQA/<tag>` predictions via the same loader the
comparison scripts use, so the correctness flag here is the same `rog_eval_hit`
those tables report.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runs import ROG_CACHE_DIR, load_eval_utils, rog_answers  # noqa: E402

WIDTH = 78


def resolve_run(run: str) -> Path:
    base = Path(run)
    if not base.is_dir():
        base = ROG_CACHE_DIR / run
    if not base.is_dir():
        raise SystemExit(f"{base}: no such run directory")
    return base


def resolve_tag(base: Path, policy: str) -> tuple[str, dict]:
    """Tag directory for `policy`, plus its summary row (threshold, model, ...).

    Policy directories carry the threshold in their name (`semantic_lfu_t0.5`),
    so the tag is read off summary.json rather than guessed; a run without a
    summary falls back to a prefix match on the stage-1 directory names.
    """
    summary = base / "summary.json"
    if summary.exists():
        rows = [r for r in json.loads(summary.read_text()) if r.get("policy") == policy]
        if rows:
            return rows[-1]["tag"], rows[-1]
    stage1 = base / "gen_rule_path"
    cands = sorted(d.name for d in stage1.iterdir()
                   if d.is_dir() and d.name.startswith(policy)) if stage1.is_dir() else []
    if not cands:
        avail = sorted({r.get("policy") for r in json.loads(summary.read_text())}
                       ) if summary.exists() else []
        raise SystemExit(f"{base}: no {policy!r} stage-1 output"
                         + (f" (have: {', '.join(map(str, avail))})" if avail else ""))
    return cands[0], {}


def stage1_files(base: Path, tag: str) -> list[Path]:
    """Stage-1 prediction files, minus the sidecar `.metrics.jsonl` files."""
    root = base / "gen_rule_path" / tag
    return sorted(p for p in root.glob("**/predictions_*.jsonl")
                  if not p.name.endswith(".metrics.jsonl"))


def load_hits(paths: list[Path]) -> tuple[list[dict], int]:
    """Every cache hit across `paths`, plus the total number of questions seen."""
    hits: list[dict] = []
    total = 0
    for path in paths:
        dataset = path.parents[2].name  # gen_rule_path/<tag>/<dataset>/RoG/<split>
        split = path.parent.name
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                total += 1
                cache = d.get("cache") or {}
                if not cache.get("hit") or cache.get("similarity") is None:
                    continue
                hits.append({
                    "id": d["id"],
                    "dataset": dataset,
                    "split": split,
                    "question": d.get("question") or "",
                    "kind": cache.get("kind"),
                    "similarity": float(cache["similarity"]),
                    "source_question": cache.get("source_question"),
                    "served_paths": d.get("prediction") or [],
                    "ground_paths": d.get("ground_paths") or [],
                })
    return hits, total


def pick(bucket: list[dict], n: int, mode: str, *, low: bool,
         rng: random.Random) -> list[dict]:
    """`extreme` takes the most illustrative end of the band, `random` samples."""
    if mode == "random":
        return sorted(rng.sample(bucket, min(n, len(bucket))),
                      key=lambda h: h["similarity"])
    ordered = sorted(bucket, key=lambda h: h["similarity"], reverse=not low)
    return sorted(ordered[:n], key=lambda h: h["similarity"])


def fmt_list(items: list, limit: int) -> str:
    """One-line join, truncated -- gold answer lists run to hundreds of names."""
    if not items:
        return ""
    head = "; ".join(str(i) for i in items[:limit])
    return head + (f"  ... {len(items) - limit} more" if len(items) > limit else "")


def fmt_paths(paths: list, limit: int) -> list[str]:
    out = []
    for path in paths[:limit]:
        out.append(" -> ".join(path) if isinstance(path, list) else str(path))
    if len(paths) > limit:
        out.append(f"... {len(paths) - limit} more")
    return out or ["(none)"]


def show(hit: dict, answers: dict, eu, path_limit: int,
         answer_limit: int) -> None:
    print("-" * WIDTH)
    print(f"  sim {hit['similarity']:.4f}   {hit['kind']}   "
          f"{hit['id']}  [{hit['dataset']} {hit['split']}]")
    print(f"  asked   : {hit['question']}")
    print(f"  matched : {hit['source_question']}")
    print("  served relation paths (planner skipped):")
    for line in fmt_paths(hit["served_paths"], path_limit):
        print(f"      {line}")
    if hit["ground_paths"]:
        print("  gold relation paths:")
        for line in fmt_paths(hit["ground_paths"], path_limit):
            print(f"      {line}")
    rec = answers.get(hit["id"])
    if rec is None:
        return
    correct = eu.rog_eval_hit(rec["pred"], rec["gold"])
    print(f"  answer  : {fmt_list(rec['pred'], answer_limit) or '(empty)'}")
    print(f"  gold    : {fmt_list(rec['gold'], answer_limit) or '(none)'}")
    print(f"  hits@1  : {'yes' if correct else 'NO'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="rog_wrong",
                    help="run tag under artifacts/rog_cache, or a path "
                         "(default: rog_wrong, the t=0.5 run)")
    ap.add_argument("--policy", default="semantic_lfu",
                    help="cache policy whose hits to sample (default: semantic_lfu)")
    ap.add_argument("-n", "--num", type=int, default=3,
                    help="hits to show per band (default: 3)")
    ap.add_argument("--low-max", type=float, default=0.60,
                    help="upper bound of the low band, exclusive (default: 0.60)")
    ap.add_argument("--high-min", type=float, default=0.90,
                    help="lower bound of the high band, inclusive (default: 0.90)")
    ap.add_argument("--sample", default="extreme", choices=["extreme", "random"],
                    help="'extreme' takes the lowest/highest similarities, "
                         "'random' samples the band (default: extreme)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --sample random")
    ap.add_argument("--path-limit", type=int, default=4,
                    help="relation paths printed per hit (default: 4)")
    ap.add_argument("--answer-limit", type=int, default=6,
                    help="answers printed per hit; gold lists run long "
                         "(default: 6)")
    ap.add_argument("--json", type=Path, help="also write the shown hits as JSON")
    args = ap.parse_args()

    if args.low_max > args.high_min:
        raise SystemExit(f"--low-max {args.low_max} overlaps --high-min "
                         f"{args.high_min}: the bands would share hits")

    base = resolve_run(args.run)
    tag, meta = resolve_tag(base, args.policy)
    paths = stage1_files(base, tag)
    if not paths:
        raise SystemExit(f"{base / 'gen_rule_path' / tag}: no predictions_*.jsonl")

    hits, total = load_hits(paths)
    threshold = meta.get("similarity_threshold")

    print(f"\nrun {base.name}  policy {args.policy}  tag {tag}")
    if threshold is not None:
        print(f"  threshold t={threshold:g}"
              + (f"   model {meta['model']}" if meta.get("model") else ""))
    print(f"  {len(hits)} cache hits over {total} questions "
          f"({100 * len(hits) / total:.1f}%)" if total else "  no questions")
    if not hits:
        raise SystemExit("no cache hits recorded in this run")
    sims = [h["similarity"] for h in hits]
    print(f"  similarity range {min(sims):.4f} .. {max(sims):.4f}")

    rng = random.Random(args.seed)
    bands = [
        (f"similarity < {args.low_max:g}", True,
         [h for h in hits if h["similarity"] < args.low_max]),
        (f"similarity >= {args.high_min:g}", False,
         [h for h in hits if h["similarity"] >= args.high_min]),
    ]

    _, answers, _ = rog_answers(base, args.policy)
    eu = load_eval_utils() if answers else None
    if not answers:
        print("  [note] no stage-2 answers found; showing served paths only")

    shown: list[dict] = []
    for title, low, bucket in bands:
        print("\n" + "=" * WIDTH)
        print(f"{title}   ({len(bucket)} of {len(hits)} hits)")
        print("=" * WIDTH)
        if not bucket:
            if low and threshold is not None and threshold >= args.low_max:
                print(f"  none -- the run's threshold t={threshold:g} rejects "
                      f"anything below {args.low_max:g}")
            else:
                print("  none in this band")
            continue
        for hit in pick(bucket, args.num, args.sample, low=low, rng=rng):
            show(hit, answers, eu, args.path_limit, args.answer_limit)
            rec = answers.get(hit["id"])
            shown.append({**hit, "band": "low" if low else "high",
                          "answer": rec["pred"] if rec else None,
                          "gold_answers": rec["gold"] if rec else None,
                          "hits1": eu.rog_eval_hit(rec["pred"], rec["gold"])
                          if rec else None})
        print("-" * WIDTH)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"run": base.name, "policy": args.policy, "tag": tag,
             "similarity_threshold": threshold, "n_hits": len(hits),
             "n_questions": total, "hits": shown}, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
