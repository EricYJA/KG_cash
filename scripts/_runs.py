"""Shared loading + scoring for the run-comparison scripts.


Standard library only, like the runners: these start under whatever interpreter
is at hand.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "compare_results"
TOG_PARAM_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "parametric_results"
ROG_CACHE_DIR = REPO_ROOT / "artifacts" / "rog_cache"
ROG_PARAM_DIR = REPO_ROOT / "artifacts" / "rog_parametric"
EVAL_UTILS = REPO_ROOT / "src" / "ToG-cache" / "eval" / "utils.py"

BRACES = re.compile(r"\{([^{}]*)\}")


def load_eval_utils():
    """ToG's own rog_eval_hit / rog_eval_f1 / exact_match, imported by path.

    Only the pure metric helpers are used -- never clean_results, which is the
    buggy one. Importing rather than reimplementing keeps these numbers on the
    same scale as eval.py's.
    """
    spec = importlib.util.spec_from_file_location("tog_eval_utils", EVAL_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tog_prediction(record: dict) -> list[str]:
    """ToG's answer: last braced span that is not the Yes/No marker, else prose.

    ToG's prompt_evaluate few-shots produce `A: {Yes}. ... the answer is {X}.`,
    and eval.py's clean_results scans to the FIRST brace pair -- returning the
    sufficiency marker. Taking the last non-Yes/No span recovers the answer;
    empty `{}` is skipped because "" is a substring of every gold string.
    """
    spans = [s.strip() for s in BRACES.findall(record.get("results") or "")]
    spans = [s for s in spans if s and s.lower() not in ("yes", "no")]
    text = spans[-1] if spans else (record.get("results") or "")
    return [p.strip() for p in text.split("\n") if p.strip()] or [text]


def rog_prediction(record: dict) -> list[str]:
    """RoG emits a newline-separated list already; no brace convention."""
    return [p.strip() for p in (record.get("prediction") or "").split("\n") if p.strip()]


def index_jsonl(path: Path, extract) -> tuple[dict[str, dict], int]:
    """{id: {gold, pred}} plus a duplicate count; a repeated id keeps its LAST.

    These files are appended to on resume, and a resume that re-answers rather
    than skips leaves an id twice with different predictions (RoG's parametric
    run has 2835 lines for 1628 questions). Keying by id de-duplicates; scoring
    raw lines would weight those questions twice.
    """
    out: dict[str, dict] = {}
    dupes = 0
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["id"] in out:
                dupes += 1
            out[d["id"]] = {"gold": d.get("ground_truth") or [],
                            "pred": extract(d), "raw": d}
    return out, dupes


def tog_answers(tag_or_dir, policy: str) -> tuple[Path | None, dict, int]:
    """Load one ToG policy's answers: compare_results/<tag>/<policy>.jsonl."""
    base = Path(tag_or_dir)
    if not base.is_dir():
        base = COMPARE_DIR / str(tag_or_dir)
    path = base / f"{policy}.jsonl"
    if not path.exists():
        return None, {}, 0
    recs, dupes = index_jsonl(path, tog_prediction)
    return path, recs, dupes


def rog_answers(tag_or_dir, policy: str, *,
                quiet: bool = False) -> tuple[Path | None, dict, int]:
    """Load one RoG policy's predictions.

    RoG's policy directories carry the similarity threshold in their name
    (`semantic_lfu_t0.90`, `semantic_lfu_t0.5`), so an exact policy name is
    matched first and a prefix match is accepted after -- otherwise comparing a
    0.5-threshold run against a 0.90 one would need the caller to spell both.

    A policy can hold MORE THAN ONE predictions.jsonl, because the split is part
    of the path: re-running a tag at `--limit 400` leaves the earlier full-split
    file behind, so `KGQA/none/` ends up with both `test/` and `test[:400]/`.
    Those are one configuration asked a different number of questions, not two
    configurations, so they are MERGED oldest-first: every question either run
    answered is available to score, and a question both answered keeps the newer
    run's answer. Keeping only the newest file instead would drop questions that
    nothing supersedes -- rog_live_virt_gemini_2 holds 387 under `test[:400]`
    plus 4 that only `test` answered, and its `_3` replicate 381 plus 75.

    Merging is within one policy directory, never across them: a prefix match
    can turn up `semantic_lfu_t0.90` beside `semantic_lfu_t0.5`, and those are
    genuinely different configurations, so the first match still wins outright.
    """
    base = Path(tag_or_dir)
    if not base.is_dir():
        base = ROG_CACHE_DIR / str(tag_or_dir)
    kgqa = base / "KGQA"
    if not kgqa.is_dir():
        return None, {}, 0
    exact = kgqa / policy
    candidates = [exact] if exact.is_dir() else sorted(
        d for d in kgqa.iterdir() if d.is_dir() and d.name.startswith(policy))
    for cand in candidates:
        preds = sorted(cand.glob("**/predictions.jsonl"),
                       key=lambda p: p.stat().st_mtime)
        if not preds:
            continue
        recs: dict[str, dict] = {}
        dupes = 0
        superseded = 0
        parts = []
        for pred in preds:
            part, part_dupes = index_jsonl(pred, rog_prediction)
            dupes += part_dupes
            superseded += len(recs.keys() & part.keys())
            parts.append(f"{pred.parent.parent.name} ({len(part)})")
            recs.update(part)
        if len(preds) > 1 and not quiet:
            print(f"  [note] {base.name}/{cand.name}: merged {len(preds)} "
                  f"prediction files -- {', '.join(parts)} -- into {len(recs)} "
                  f"questions ({superseded} answered more than once, "
                  "newest kept)")
        return preds[-1], recs, dupes
    return None, {}, 0


def load_answers(system: str, tag, policy: str):
    return (tog_answers if system == "tog" else rog_answers)(tag, policy)


def rog_universe(tag_or_dir, policy: str, split: str) -> set[str]:
    """The question ids stage 1 was ASKED for `split` -- a run's real denominator.

    Stage 2 writes no record at all when the reasoner's LLM call raises:
    predict_answer_api.py catches it, warns, and returns None, which upstream
    treats as "no prediction" and skips. Its predictions.jsonl is therefore
    silently shorter than the split, and scoring it over its own ids reports
    accuracy on the questions the run survived. Stage 1's file holds the whole
    split, so it supplies the ids those failures should be counted against.

    Split is matched exactly rather than globbed, because a run's directory name
    (`test[:400]`) is itself a glob character class.
    """
    base = Path(tag_or_dir)
    if not base.is_dir():
        base = ROG_CACHE_DIR / str(tag_or_dir)
    stage1 = base / "gen_rule_path"
    if not stage1.is_dir():
        return set()
    exact = stage1 / policy
    candidates = [exact] if exact.is_dir() else sorted(
        d for d in stage1.iterdir() if d.is_dir() and d.name.startswith(policy))
    for cand in candidates:
        ids: set[str] = set()
        for path in cand.rglob("predictions_*.jsonl"):
            if path.name.endswith(".metrics.jsonl") or path.parent.name != split:
                continue
            with path.open() as fh:
                ids |= {json.loads(line)["id"] for line in fh if line.strip()}
        if ids:
            return ids
    return set()


def score(records: dict, qids, eu) -> dict:
    """Hits@1 / F1 / precision / recall over `qids`, plus the per-question hits.

    A qid with no record scores ZERO on every metric instead of being dropped.
    Callers that pass an intersection never reach that branch; callers that pass
    a fixed universe (average_replicates.py --universe) are asking for exactly
    it, because there a question the run never answered is a question it failed.
    """
    hits = f1s = precs = recs = 0.0
    per_question = {}
    for qid in qids:
        rec = records.get(qid)
        if rec is None:
            per_question[qid] = 0.0
            continue
        h = eu.rog_eval_hit(rec["pred"], rec["gold"])
        f1, p, r = eu.rog_eval_f1(rec["pred"], rec["gold"])
        hits += h
        f1s += f1
        precs += p
        recs += r
        per_question[qid] = h
    n = len(qids) or 1
    return {"hits1": 100 * hits / n, "f1": 100 * f1s / n,
            "precision": 100 * precs / n, "recall": 100 * recs / n,
            "per_question": per_question}


def jsonl_ids(path: Path) -> list[str]:
    """Question ids in the order a file records them; a repeat keeps its FIRST place."""
    seen: set[str] = set()
    out: list[str] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            qid = json.loads(line)["id"]
            if qid not in seen:
                seen.add(qid)
                out.append(qid)
    return out


def order_files(system: str, tag_or_dir, policy: str) -> list[Path]:
    """Every file that records the order `policy` was asked its questions in.

    RoG's stage 1 is preferred over stage 2 because it holds the whole split --
    stage 2 is missing whatever the reasoner crashed on -- but a run with no
    stage 1 on disk still yields its answer order.
    """
    base = Path(tag_or_dir)
    if system == "tog":
        if not base.is_dir():
            base = COMPARE_DIR / str(tag_or_dir)
        path = base / f"{policy}.jsonl"
        return [path] if path.exists() else []
    if not base.is_dir():
        base = ROG_CACHE_DIR / str(tag_or_dir)
    files: list[Path] = []
    for stage, pattern in (("gen_rule_path", "predictions_*.jsonl"),
                           ("KGQA", "predictions.jsonl")):
        root = base / stage
        if not root.is_dir():
            continue
        exact = root / policy
        candidates = [exact] if exact.is_dir() else sorted(
            d for d in root.iterdir() if d.is_dir() and d.name.startswith(policy))
        for cand in candidates:
            files += [p for p in sorted(cand.rglob(pattern))
                      if not p.name.endswith(".metrics.jsonl")]
        if files:
            return files
    return files


def ask_order(system: str, tag_or_dir, policy: str) -> list[str]:
    """The order this run asked its questions in: its LONGEST single file's order.

    The longest file rather than a merge of them, because a policy directory can
    hold several splits (see `rog_answers`) and the leftovers are not always
    prefixes of each other -- rog_live_virt_gemini_2 keeps a 7-line `test` file
    beside its 400-question `test[:400]` one, and concatenating them would seat
    those 7 ids ahead of questions that were really asked first. One file is one
    uninterrupted pass over the split, so its order is the run's order.
    """
    orders = [jsonl_ids(p) for p in order_files(system, tag_or_dir, policy)]
    return max(orders, key=len) if orders else []


def is_subsequence(short: list[str], long: list[str]) -> bool:
    """True if `short` appears inside `long` in the same relative order."""
    it = iter(long)
    return all(qid in it for qid in short)
