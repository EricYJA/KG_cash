#!/usr/bin/env python3
"""The parametric experiment: the LLM answering the benchmark on its own.

    question  ->  answer        (no KG, no retrieval, no RAG, no cache, no tools)

ONE run, ONE prompt, ONE extraction rule, ONE scorer, and the result is the
floor that BOTH the RoG and the ToG experiments are read against. That is the
point of this script existing next to run_rog_parametric_experiment.py and
run_tog_parametric_experiment.py: those two ablate their own pipeline, so each
inherits its parent's answer format, and their numbers are therefore NOT
comparable to each other. Measured on the 1628 questions they share, with the
same model, the same gold and the same scorer, they disagreed by 5.5 points of
Hits@1 (RoG 61.3, ToG 55.8) purely because RoG's reasoner prompt returns a list
of candidates (~4 per question) while ToG's few-shot prompt commits to one
braced answer (~1.4). Both are honest measurements of their own pipeline's
no-KG behaviour; neither is "what the model knows". This is that number.

WHAT IS SWITCHED OFF, and how it is switched off:

  * No knowledge graph. No SPARQL endpoint is contacted, no subgraph is loaded,
    no relation path is planned. This script has no KG code path to disable --
    it never had one.
  * No retrieval / RAG. Nothing is prepended to the prompt but the question.
  * No provider-side grounding. llm_client.complete_json posts a plain
    OpenAI-compatible chat-completions body -- `model`, `messages`, `stream`,
    and `temperature` only. There is no `tools`, no `tool_choice`, no
    `web_search_options`, and no Google `google_search` / `googleSearchRetrieval`
    field, on any vendor. Gemini is reached over its OpenAI-compatible endpoint,
    where grounding must be requested explicitly and never is. So the weights
    are the only knowledge source, by construction rather than by flag.
  * No cache. There is nothing to cache: a cache in these experiments stores
    planned relation paths (RoG) or a traversal chain (ToG), and neither exists
    here. One LLM call per question, always.

QUESTIONS. Read from ToG's data files, which are supersets of both systems'
splits -- WebQSP 1639 >= RoG-webqsp test 1628, CWQ 3531 >= RoG-cwq test 3052 --
so a single run covers every question either system will be compared on.

GOLD ANSWERS. Recorded per question for self-scoring, but note that on CWQ the
two systems do not agree on them: ToG's cwq.json keeps one answer string per
question, RoG-cwq keeps the alias list, and they match on only 2211 of the 3052
shared questions (WebQSP's agree on all 1628). So the gold stored here is ToG's,
and scripts/compare_parametric_vs_live.py scores BOTH sides of a comparison
against the gold of the live run it is comparing to -- which is the only way the
delta stays a measure of the KG rather than of two answer keys.

Restartable: re-running the same --run-tag resumes (answered questions are
skipped) and re-scores; --fresh wipes the tag and starts over. The tag's model
and dataset are pinned on first use, so a resume can never mix two models'
answers into one file.

    ./scripts/run_parametric_experiment.py                      # webqsp, 20 q
    ./scripts/run_parametric_experiment.py --limit all
    ./scripts/run_parametric_experiment.py --dataset cwq --limit all
    ./scripts/run_parametric_experiment.py --vendor google --env-file .env_gemini \
        --run-tag gemini --limit all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# llm_client / llm_config / cache_metrics live with ToG's code but are
# system-agnostic; imported rather than copied so this run's HTTP behaviour, key
# rotation and timing sidecar are byte-identical to every other experiment's.
sys.path.insert(0, str(REPO_ROOT / "src" / "ToG-cache" / "ToG"))

from _ablation import (guard_run_identity,  # noqa: E402
                       resolve_model, warn_tag_model_mismatch)
from _runs import load_eval_utils, rog_prediction, score  # noqa: E402
from _tog_common import (add_env_file_arg, load_dotenv, load_env_keys,  # noqa: E402
                         preload_dotenv, KEYS_PATH)

try:
    from cache_metrics import (aggregate_run_metrics,  # noqa: E402
                               append_question_metrics, metrics_sidecar_path)
    from llm_client import ChatMessage, LLMChatClient  # noqa: E402
    from llm_client import LLMKeyPoolExhaustedError  # noqa: E402
    from llm_config import resolve_llm_config  # noqa: E402
except ImportError as exc:  # httpx is the only third-party dependency here
    raise SystemExit(
        f"cannot import the LLM client ({exc}). Run this under a Python that has "
        f"httpx installed -- e.g. `conda run -n KG_cash python "
        f"scripts/run_parametric_experiment.py ...`"
    ) from exc

CONDITION = "parametric"
OUT_ROOT = REPO_ROOT / "artifacts" / "parametric"
DATA_DIR = REPO_ROOT / "src" / "ToG-cache" / "data"

# See MAX_CONSECUTIVE_FAILURES in main_freebase.py: scattered failures are the
# vendor having a bad minute and are recorded unanswered; this many in a row is
# something being down, and continuing only fills the output with empty answers.
MAX_CONSECUTIVE_FAILURES = 20


# The whole prompt. It licenses parametric knowledge (there is nothing else to
# answer from) and pins the answer FORMAT, because the format is what made the
# two per-system ablations incomparable in the first place: RoG's scorer divides
# precision by the number of predicted lines, so a paragraph craters precision
# and a single guess craters recall on WebQSP's ~10-answer gold sets. One entity
# per line is the format both systems' scorers read natively.
#
# The abstention clause is deliberate. Without it the model guesses on entities
# it has no memory of, and the experiment reports the vendor's appetite for
# guessing rather than what it actually knows.
SYSTEM_PROMPT = (
    "You answer questions about real-world entities from your own knowledge. "
    "No reasoning paths, knowledge-graph triples or retrieved documents are "
    "provided; answer from what you already know. "
    "Output ONLY the answer(s), one entity name per line. No numbering, no "
    "bullets, no markdown, no preamble, no explanation. If you do not know the "
    "answer, output nothing."
)

USER_TEMPLATE = (
    "Please answer the following question. Keep each answer as simple as "
    "possible, and return all the possible answers, one per line.\n\n"
    "Question:\n{question}"
)

# Lines that are framing rather than an answer entity. Anchored and specific so a
# real entity ("The Beatles", "Answer Man") survives: only sentence lead-ins are
# dropped. Same rule as RoG's predict_answer_api._extract_answers, so answers
# written here are shaped exactly like the ones the grounded runs write.
_FRAMING_RE = re.compile(
    r"^(based on|according to|here (are|is)\b|the answer\b|the following\b|"
    r"answer[:\s]|answers[:\s]|these are\b|from the reasoning|the reasoning|"
    r"no answer\b|not (contain|available)|there (is|are) no\b|note:)",
    re.IGNORECASE,
)


def extract_answers(text: str) -> list[str]:
    """Reduce a completion to bare answer entities, one per line.

    Conservative on purpose: strips list/markdown decoration and drops obvious
    framing lines, but never tries to guess entities out of a paragraph. With
    SYSTEM_PROMPT the model already returns a clean list; this repairs the
    occasional stray bullet so precision is not taxed for it.
    """
    answers = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*•]\s+", "", line)        # bullet markers
        line = re.sub(r"^\d+[\.\)]\s+", "", line)      # "1." / "1)" numbering
        line = line.replace("**", "").replace("__", "").strip()
        if not line or line.endswith(":"):             # empty, or a lead-in
            continue
        if _FRAMING_RE.match(line):
            continue
        answers.append(line)
    return answers


def _webqsp_gold(row: dict) -> list[str]:
    """WebQSP gold names, mirroring utils.extract_ground_truth / eval align()."""
    out = []
    for parse in row.get("Parses") or []:
        for ans in parse.get("Answers") or []:
            name = ans.get("EntityName") or ans.get("AnswerArgument")
            if name is not None:
                out.append(str(name))
    return out


def _cwq_gold(row: dict) -> list[str]:
    """CWQ gold, mirroring utils.extract_ground_truth."""
    raw = row.get("answers", row.get("answer"))
    out = []
    if isinstance(raw, str):
        out.append(raw)
    elif isinstance(raw, list):
        for ans in raw:
            if isinstance(ans, dict):
                out.extend(str(a) for a in (ans.get("aliases") or []))
                if ans.get("answer"):
                    out.append(str(ans["answer"]))
            elif ans is not None:
                out.append(str(ans))
    return out


# dataset -> (file, id field, question field, gold extractor). Deliberately the
# same fields ToG's utils.extract_record_id / extract_ground_truth read, so the
# ids here join against both systems' output files without translation.
DATASETS = {
    "webqsp": ("WebQSP.json", "QuestionId", "RawQuestion", _webqsp_gold),
    "cwq": ("cwq.json", "ID", "question", _cwq_gold),
}


def load_questions(dataset: str) -> list[tuple[str, str, list[str]]]:
    """(id, question, gold) for every row of a dataset, in file order."""
    try:
        filename, id_key, q_key, gold_of = DATASETS[dataset]
    except KeyError:
        raise SystemExit(
            f"unknown --dataset {dataset!r}; choose from "
            f"{sorted(DATASETS)}") from None
    path = DATA_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing dataset file {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [(str(r[id_key]), str(r[q_key]), gold_of(r)) for r in rows]


def parse_limit(value):
    """argparse type for --limit: a positive int, or 'all' (-> None)."""
    text = str(value or "").strip().lower()
    if text in ("", "all", "none"):
        return None
    try:
        limit = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid --limit {value!r}: expected a positive integer or 'all'"
        ) from None
    if limit <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid --limit {value!r}: expected a positive integer or 'all'")
    return limit


def answered_ids(path: Path) -> set[str]:
    """Question ids already recorded in an output file, for resuming."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    # argparse evaluates each default as the argument is declared, so the env
    # file has to be loaded before the parser is built or its values are ignored
    # (and `source .env` does not export them into a child process either).
    preload_dotenv()
    env = os.environ.get
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_file_arg(p)
    p.add_argument("--dataset", default=env("DATASET", "webqsp"),
                   help=f"{' | '.join(sorted(DATASETS))} (default: webqsp)")
    p.add_argument("-n", "--limit", type=parse_limit, default=env("N", "20"),
                   help="questions to ask, or 'all'")
    p.add_argument("--vendor", default=env("VENDOR", "tamu"),
                   help="tamu | openai | google")
    p.add_argument("--model", default=env("MODEL", ""),
                   help="override the vendor's default model id")
    p.add_argument("--run-tag", default=env("RUN_TAG", ""),
                   help="subdirectory under artifacts/parametric/. One tag per "
                        "model: the tag's model is pinned on first use.")
    p.add_argument("--fresh", action="store_true", default=env("FRESH", "0") == "1",
                   help="wipe this run-tag's outputs and start over (default: resume)")
    p.add_argument("--timeout-s", type=float, default=120.0,
                   help="per-request HTTP timeout")
    args = p.parse_args()

    # .env_keys holds a pool the client rotates through on a 400/5xx; with a pool
    # present the single LLM_API_KEY in .env is optional.
    n_keys = load_env_keys()
    load_dotenv(required=() if n_keys else ("LLM_API_KEY",))
    if n_keys:
        print(f"[keys] {n_keys} API keys from {KEYS_PATH.name}; the client falls "
              f"back to the next one on a 400/5xx")

    out_dir = OUT_ROOT / args.run_tag if args.run_tag else OUT_ROOT
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{CONDITION}.jsonl"
    metrics_path = Path(metrics_sidecar_path(str(out_jsonl)))

    model = resolve_model(args.vendor, args.model)
    warn_tag_model_mismatch(args.run_tag, model)
    # The dataset is part of the identity as well as the model: two datasets in
    # one file would be scored as one run over a question set that never existed.
    guard_run_identity(out_dir, {"dataset": args.dataset, "vendor": args.vendor,
                                 "model": model}, fresh=args.fresh)

    questions = load_questions(args.dataset)
    if args.limit is not None:
        questions = questions[:args.limit]
    done = answered_ids(out_jsonl)
    todo = [q for q in questions if q[0] not in done]

    print("\n" + "=" * 72)
    print(f">>> PARAMETRIC experiment  [{args.dataset}, "
          f"N={len(questions)}, {args.vendor}/{model}"
          f"{' (vendor default)' if not args.model else ''}]")
    print(">>> the model gets the question and nothing else: no KG, no retrieval,")
    print(">>> no tools, no provider grounding, no cache. One LLM call each.")
    if done:
        print(f">>> resuming: {len(done)} already answered, {len(todo)} to go")
    print("=" * 72)

    config = resolve_llm_config(vendor=args.vendor, model=args.model or None)
    client = LLMChatClient(config, timeout_s=args.timeout_s)

    failed = 0                # questions recorded unanswered after an LLM failure
    consecutive_failures = 0  # trips MAX_CONSECUTIVE_FAILURES when the run is broken
    for i, (qid, question, gold) in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            text = client.complete_json(
                [ChatMessage(role="system", content=SYSTEM_PROMPT),
                 ChatMessage(role="user", content=USER_TEMPLATE.format(question=question))],
                temperature=0.0,
            )
            error = None
        except LLMKeyPoolExhaustedError as exc:
            # Every key was tried for this one request and every one failed, so
            # nothing is left to answer the next question with. Stop before this
            # question is written; the finished ones stay and the run resumes.
            raise SystemExit(
                f"\nEvery API key failed on one request, so no further question "
                f"can be answered -- stopping rather than filling the output with "
                f"empty answers. Re-run with the same --run-tag to resume."
                f"\nCause: {exc}") from exc
        except Exception as exc:
            # The client has already retried and rotated keys. Record the
            # question unanswered (it scores as wrong, keeping the denominator
            # the whole split) and carry on.
            failed += 1
            consecutive_failures += 1
            print(f"[warn] {qid} failed, recording it unanswered: {exc}", flush=True)
            text, error = "", str(exc)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(
                    f"{consecutive_failures} questions failed in a row -- stopping "
                    f"rather than filling the output with empty answers. "
                    f"Last error: {exc}") from exc
        else:
            consecutive_failures = 0

        prediction = "\n".join(extract_answers(text))
        with out_jsonl.open("a") as fh:
            # `prediction` is RoG's field name and `results` is ToG's, holding
            # the same text: scripts/_runs.py reads this one file with either
            # system's extractor and gets the same answer list back, which is
            # what makes one run serve both comparisons.
            fh.write(json.dumps({
                "id": qid, "question": question,
                "prediction": prediction, "results": prediction,
                "ground_truth": gold,
                # Every KG run stores the triples it reasoned over here; this one
                # has none, and the empty list is the record of that.
                "reasoning_chains": [],
                **({"error": error} if error else {}),
            }) + "\n")
        append_question_metrics(str(metrics_path), {
            "id": qid, "question": question,
            # No cache exists in this condition; recorded as a miss so
            # aggregate_run_metrics counts every question's full cost.
            "cache_hit": False, "cache_hit_type": None,
            "elapsed_s": time.perf_counter() - t0, "llm_calls": 1,
            **({"failed": True, "error": error} if error else {}),
        })
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {qid}", flush=True)

    if failed:
        print(f"WARNING: {failed} questions were recorded unanswered because their "
              f"LLM calls failed; they score as wrong.")

    # ---- score -------------------------------------------------------------
    # rog_prediction is the plain newline split: the framing/bullet cleanup
    # already happened at write time, so reading is a split and nothing more.
    eu = load_eval_utils()
    records, dupes = {}, 0
    with out_jsonl.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] in records:
                dupes += 1
            records[rec["id"]] = {"gold": rec.get("ground_truth") or [],
                                  "pred": rog_prediction(rec)}
    if dupes:
        print(f"[note] {dupes} duplicate id(s) in {out_jsonl.name} "
              f"(kept the last answer for each)")

    qids = [q[0] for q in questions]
    metrics = score(records, qids, eu)
    metrics.pop("per_question", None)
    timing, run_summary, _breakdown, _per_loop = aggregate_run_metrics(str(metrics_path))

    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "condition": CONDITION,
        "dataset": args.dataset,
        "vendor": args.vendor,
        "model": model,
        "questions": len(qids),
        "answered": len(records),
        "unanswered_after_failure": failed,
        **{k: round(v, 4) for k, v in metrics.items()},
        "timing": timing,
        "run": run_summary,
        "answers": str(out_jsonl),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{CONDITION}.done").write_text("")

    print("\n" + "=" * 72)
    print(f">>> PARAMETRIC  {args.dataset}  {model}   n={len(qids)}")
    print("=" * 72)
    for key in ("hits1", "f1", "precision", "recall"):
        print(f"  {key:<10}{metrics[key]:>8.2f}")
    print(">>> hits1 / f1 are the numbers the comparison tables use. precision and")
    print(">>> recall are RoG's evaluate_results definition (matched golds divided by")
    print(">>> predicted lines), so precision can exceed 100 where a gold list repeats")
    print(">>> an answer; that is upstream's metric, kept for comparability.")
    print(f"\nwrote {out_dir / 'summary.json'} and {out_jsonl}")
    print(">>> compare it against the KG-grounded runs with "
          "scripts/compare_parametric_vs_live.py")


if __name__ == "__main__":
    main()
