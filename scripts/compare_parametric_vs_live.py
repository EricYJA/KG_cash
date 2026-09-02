#!/usr/bin/env python3
"""Does the live KG earn its keep? Parametric (no-KG) vs live-KG runs, Gemini.

The parametric ablations answer each question from the model's own memory with
no graph at all; the live runs answer it with paths retrieved from Freebase over
SPARQL. Subtracting the two isolates what the KG actually contributed -- which is
the question worth asking on this data, because the loaded dump carries labels
for only ~0.08% of its entities, so most retrieved paths terminate in
`UnName_Entity` and both systems fall back to parametric memory anyway.

The parametric side is ALWAYS the shared run from
scripts/run_parametric_experiment.py (artifacts/parametric/<tag>/parametric.jsonl):
one prompt, one extraction rule, one answer set, used as the single floor for
every system and backend. The older per-system ablations (rog_parametric/,
parametric_results/) are deliberately not read -- their answer format is their
own pipeline's (RoG's reasoner returns a list, ToG's few-shot returns one braced
answer), so each was comparable to its own grounded run but not to the other's,
and two systems measured against two different floors cannot be put in one table.
Pin a particular shared run with --parametric-run <tag>.

Compared, per system and backend:

    baseline  artifacts/parametric/<tag>/parametric.jsonl   (shared, both systems)
    ToG       compare_results/tog_rerun_live_{virt,oxi}_gemini/none.jsonl
    RoG       rog_cache/rog_live_{virt,oxi}_gemini/KGQA/none/.../predictions.jsonl

`none` is the uncached policy, so the comparison is against the KG itself rather
than against a cache policy.

Runs are joined on question `id` and scored on the INTERSECTION only: the shared
baseline covers one dataset's questions and the live runs differ in length and in
dataset, so a whole-file comparison would be reading two different question sets
(a CWQ live run against a WebQSP baseline shares no ids at all and is skipped). Metrics
come from ToG's own rog_eval_hit / rog_eval_f1 (imported, not reimplemented), and
ToG answers are extracted with the same last-braced-span rule as
scripts/rescore_tog.py -- eval.py's first-span scan returns the `{Yes}`
sufficiency marker instead of the answer.

Two parametric columns, therefore. `pAll` is the baseline over ALL of its own
questions -- the full 1639 on WebQSP -- scored on its own gold: one fixed number
per baseline, the same on every row, which is the floor to quote. `param` is that
same file restricted to the questions the row's live run actually answered and
scored on the live gold, which is the only one the `delta` may be taken against.
They coincide when the live run is complete.

Beyond the headline delta it reports the swing: how many questions the KG FIXED
(parametric wrong -> live right) against how many it BROKE (parametric right ->
live wrong). A KG that is pulling its weight fixes far more than it breaks; two
similar numbers mean the delta is churn, not retrieval.

    ./scripts/compare_parametric_vs_live.py
    ./scripts/compare_parametric_vs_live.py --systems tog
    ./scripts/compare_parametric_vs_live.py --parametric-run parametric_exp
    ./scripts/compare_parametric_vs_live.py --csv artifacts/parametric_vs_live.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ablation import VENDOR_DEFAULT_MODEL, resolve_model  # noqa: E402

TAMU_DEFAULT = VENDOR_DEFAULT_MODEL["tamu"]

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE_DIR = REPO_ROOT / "src" / "ToG-cache" / "output" / "compare_results"
ROG_CACHE_DIR = REPO_ROOT / "artifacts" / "rog_cache"
PARAM_DIR = REPO_ROOT / "artifacts" / "parametric"
EVAL_UTILS = REPO_ROOT / "src" / "ToG-cache" / "eval" / "utils.py"

BRACES = re.compile(r"\{([^{}]*)\}")

def _model_from(blob, vendor_default) -> str | None:
    """Pull a model id out of whatever metadata shape a run left behind.

    An empty/None model is not "no model": it means the runner fell through to
    the vendor preset. Resolving it here is what stops a Haiku run that was
    tagged `gemini_*` from being silently paired against a Gemini live run.
    """
    if not isinstance(blob, dict):
        return None
    for key in ("identity", "args"):
        inner = blob.get(key)
        if isinstance(inner, dict) and "model" in inner:
            return resolve_model(inner.get("vendor") or "tamu",
                                 inner["model"]) or vendor_default
    if "model" in blob:
        return resolve_model(blob.get("vendor") or "tamu", blob["model"])
    return None


def model_of(source: Path) -> str:
    """Best-effort model id for a run, resolved through the vendor preset.

    Each stage writes its config somewhere different: the parametric runners now
    pin run_config.json (scripts/_ablation.guard_run_identity), ToG's cache runs
    leave run_config.json / summary.json beside the answers, and RoG's
    containerised stages leave args.txt and run_config.json in the predictions
    directory. Try them in that order.
    """
    base = source if source.is_dir() else source.parent
    candidates = [base / "run_config.json", base / "summary.json",
                  base.parent / "run_config.json",
                  base.parent / "manifest_parametric_no_kg.json"]
    candidates += sorted(base.glob("**/run_config.json"))[:1]
    candidates += sorted(base.glob("**/args.txt"))[:1]
    for path in candidates:
        if not path.exists():
            continue
        try:
            model = _model_from(json.loads(path.read_text()), TAMU_DEFAULT)
        except (json.JSONDecodeError, OSError):
            continue
        if model:
            return model
    return "unknown"


def short_model(model: str) -> str:
    """A legend-sized name for a model id, e.g. 'Gemini', 'Haiku'."""
    m = model.lower()
    for needle, label in (("gemini", "Gemini"), ("haiku", "Haiku"),
                          ("sonnet", "Sonnet"), ("opus", "Opus"),
                          ("gpt", "GPT")):
        if needle in m:
            return label
    return model.split(".")[-1][:12] or "unknown"


def backend_of(tag: str) -> str:
    """'Virt'/'Oxi' from a run tag; 'virt' is this repo's short form."""
    t = tag.lower()
    return "Virt" if "virt" in t else "Oxi" if "oxi" in t else "?"


def _questions_in(path: Path) -> int:
    """Distinct question ids in a shared parametric file, for tie-breaking."""
    try:
        with path.open() as fh:
            return len({json.loads(line)["id"] for line in fh if line.strip()})
    except (json.JSONDecodeError, OSError, KeyError):
        return 0


def discover_parametric(run_tag: str = "") -> list[tuple[str, Path, str]]:
    """(tag, answers path, model) for every run of the SHARED parametric experiment.

    This is the only baseline the comparison uses. scripts/run_parametric_experiment.py
    writes one file per run tag holding both `prediction` (RoG's field) and
    `results` (ToG's), with the same text in each, so either system's extractor
    reads it and returns the same answer list -- which is what lets a single run
    be the floor for both systems, and what the per-system ablations could never
    be.

    Ordered widest-first within a model, so a live run pairs with the fullest
    baseline that model has rather than with whichever tag happens to sort first
    (the smoke-test tags in artifacts/parametric/ cover a few dozen questions;
    a real run covers the split).
    """
    out = []
    roots = [PARAM_DIR] + (sorted(PARAM_DIR.iterdir()) if PARAM_DIR.is_dir() else [])
    for d in roots:
        f = d / "parametric.jsonl"
        if d.is_dir() and f.exists():
            tag = "parametric" if d == PARAM_DIR else d.name
            if run_tag and tag != run_tag:
                continue
            out.append((tag, f, model_of(f)))
    out.sort(key=lambda r: -_questions_in(r[1]))
    return out


def discover_live(system: str) -> list[tuple[str, Path, str]]:
    """(tag, answers path, model) for every LIVE-KG run's uncached `none` policy.

    `none` on purpose: comparing against a cache policy would fold cache effects
    into a number that is supposed to isolate the knowledge graph.
    """
    out = []
    if system == "tog":
        for d in sorted(COMPARE_DIR.iterdir()) if COMPARE_DIR.is_dir() else []:
            f = d / "none.jsonl"
            if d.is_dir() and "live" in d.name and f.exists():
                out.append((d.name, f, model_of(f)))
    else:
        for d in sorted(ROG_CACHE_DIR.iterdir()) if ROG_CACHE_DIR.is_dir() else []:
            none_dir = d / "KGQA" / "none"
            if (d.is_dir() and "live" in d.name and none_dir.is_dir()
                    and any(none_dir.glob("**/predictions.jsonl"))):
                out.append((d.name, none_dir, model_of(none_dir)))
    return out


def build_pairs(system: str, model_filter: str, allow_mismatch: bool,
                run_tag: str = ""):
    """Pair each live run with the shared parametric run that used the SAME model."""
    params = discover_parametric(run_tag)
    lives = discover_live(system)
    pairs, unmatched = [], []
    for live_tag, live_src, live_model in lives:
        if model_filter and model_filter.lower() not in live_model.lower():
            continue
        same = [p for p in params if p[2] == live_model]
        if not same and allow_mismatch:
            same = params[:1]
        if not same:
            unmatched.append((live_tag, live_model))
            continue
        param_tag, param_src, param_model = same[0]
        label = (f"{system.upper().replace('TOG', 'ToG').replace('ROG', 'RoG')} "
                 f"{short_model(live_model)} ({backend_of(live_tag)})")
        pairs.append((label, param_tag, param_src, param_model,
                      live_tag, live_src, live_model))
    return pairs, unmatched, params


def load_eval_utils():
    spec = importlib.util.spec_from_file_location("tog_eval_utils", EVAL_UTILS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tog_prediction(record: dict) -> list[str]:
    """ToG's answer: last braced span that is not the Yes/No marker, else prose."""
    spans = [s.strip() for s in BRACES.findall(record.get("results") or "")]
    spans = [s for s in spans if s and s.lower() not in ("yes", "no")]
    text = spans[-1] if spans else (record.get("results") or "")
    return [p.strip() for p in text.split("\n") if p.strip()] or [text]


def rog_prediction(record: dict) -> list[str]:
    """RoG emits a newline-separated list already; no brace convention."""
    return [p.strip() for p in (record.get("prediction") or "").split("\n") if p.strip()]


def _index(lines, extract) -> tuple[dict[str, dict], int]:
    """{id: record} plus a duplicate count; a repeated id keeps its LAST answer.

    These files are appended to on resume, and a resume that re-answers rather
    than skips leaves the same id twice with different predictions -- RoG's
    parametric run has 2835 lines for 1628 questions that way. Keying by id
    de-duplicates; scoring the raw lines would weight 74% of its questions twice.
    """
    out: dict[str, dict] = {}
    dupes = 0
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        if d["id"] in out:
            dupes += 1
        out[d["id"]] = {"gold": d.get("ground_truth") or [], "pred": extract(d)}
    return out, dupes


def load_tog(path: Path) -> tuple[dict[str, dict], int]:
    if not path.exists():
        return {}, 0
    return _index(path.open(), tog_prediction)


def load_rog(source: Path) -> tuple[dict[str, dict], int]:
    """RoG buries predictions.jsonl under KGQA/<policy>/<data>/<model>/<split>/<cfg>/.

    A plain .jsonl is taken as-is: that is how the shared parametric run arrives,
    and it uses RoG's `prediction` field so the same extractor reads it.
    """
    if source.is_file():
        return _index(source.open(), rog_prediction)
    hits = sorted(source.glob("**/predictions.jsonl"))
    if not hits:
        return {}, 0
    return _index(hits[0].open(), rog_prediction)


def full_metrics(source: dict, eu) -> tuple[float, float]:
    """Hits@1 / F1 over EVERY question in a run, not just the shared subset.

    This is the baseline's own headline number -- all 1639 WebQSP questions for
    the full parametric run -- and it stays fixed no matter which live run the
    row is about, so one figure describes the floor across the whole table. The
    per-row `param` column next to it is the same file restricted to the
    questions that row's live run actually answered; the two differ when a live
    run is partial, and only the restricted one is comparable to `live`.

    Scored against this file's OWN gold, necessarily: the questions outside the
    intersection have no live counterpart to borrow a key from. On WebQSP the
    two keys agree; on CWQ they do not (see compare()), so read this column as
    the baseline measured on its own dataset.
    """
    hits = f1s = 0.0
    for rec in source.values():
        hits += eu.rog_eval_hit(rec["pred"], rec["gold"])
        f1, _p, _r = eu.rog_eval_f1(rec["pred"], rec["gold"])
        f1s += f1
    n = len(source) or 1
    return hits / n, f1s / n


def compare(label: str, param: dict, live: dict, eu) -> dict | None:
    """Metrics for both conditions over the questions they share.

    BOTH sides are scored against the LIVE run's gold answers, never each file's
    own. The two are the same key on WebQSP, but not on CWQ: ToG's cwq.json keeps
    one answer string per question and RoG-cwq keeps the alias list, and they
    agree on only 2211 of the 3052 shared questions. Scoring each side against
    its own key there would make the delta partly a difference of answer keys
    rather than of the knowledge graph, which is the one thing it is supposed to
    measure.
    """
    common = sorted(set(param) & set(live))
    if not common:
        print(f"  [skip] {label}: no overlapping question ids "
              f"(parametric {len(param)}, live {len(live)})")
        return None

    def norm(gold):
        return {str(g).strip().lower() for g in gold}

    differing = sum(1 for q in common if norm(param[q]["gold"]) != norm(live[q]["gold"]))
    if differing:
        print(f"  [note] {label}: {differing}/{len(common)} questions carry "
              f"different gold answers in the two files; both sides scored "
              f"against the live run's gold")

    # Same answers, different list length = one file repeats a gold alias the
    # other collapsed. rog_eval_f1 counts matches over the gold LIST and divides
    # by len(prediction), so a repeated alias is counted once per copy and
    # inflates precision (46 of 1639 rows in parametric_exp score F1 > 1 this
    # way). Hits@1 is unaffected. It is why pAll and param can disagree in the
    # table even when nAll == n: they are reading two gold conventions.
    redundant = sum(1 for q in common
                    if norm(param[q]["gold"]) == norm(live[q]["gold"])
                    and len(param[q]["gold"]) != len(live[q]["gold"]))
    if redundant:
        print(f"  [note] {label}: {redundant}/{len(common)} questions list the "
              f"same gold answers a different number of times in the two files "
              f"(repeated aliases); F1 shifts with the repeats, Hits@1 does not")

    def metrics(source):
        hits = f1s = 0.0
        per_question = {}
        for qid in common:
            gold = live[qid]["gold"]
            pred = source[qid]["pred"]
            hit = eu.rog_eval_hit(pred, gold)
            f1, _p, _r = eu.rog_eval_f1(pred, gold)
            hits += hit
            f1s += f1
            per_question[qid] = hit
        n = len(common)
        return hits / n, f1s / n, per_question

    p_hit, p_f1, p_each = metrics(param)
    l_hit, l_f1, l_each = metrics(live)
    fixed = sum(1 for q in common if not p_each[q] and l_each[q])
    broke = sum(1 for q in common if p_each[q] and not l_each[q])
    a_hit, a_f1 = full_metrics(param, eu)

    return {"config": label,
            "n_parametric_all": len(param),
            "parametric_all_hits1": round(100 * a_hit, 2),
            "parametric_all_f1": round(100 * a_f1, 2),
            "n_common": len(common), "n_live": len(live),
            "parametric_hits1": round(100 * p_hit, 2),
            "live_hits1": round(100 * l_hit, 2),
            "delta_hits1": round(100 * (l_hit - p_hit), 2),
            "parametric_f1": round(100 * p_f1, 2),
            "live_f1": round(100 * l_f1, 2),
            "delta_f1": round(100 * (l_f1 - p_f1), 2),
            "kg_fixed": fixed, "kg_broke": broke, "net": fixed - broke}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="tog,rog",
                    help="comma-separated: tog, rog (default: both)")
    ap.add_argument("--model", default="",
                    help="only compare runs whose resolved model matches this "
                         "substring, e.g. --model gemini (default: all models)")
    ap.add_argument("--allow-model-mismatch", action="store_true",
                    help="pair a live run with a parametric run that used a "
                         "DIFFERENT model. The delta then mixes the model change "
                         "with the KG change and is not the KG's contribution; "
                         "every such row is flagged.")
    ap.add_argument("--parametric-run", default="",
                    help="pin the shared parametric run to use as the baseline, "
                         "by tag under artifacts/parametric/ (e.g. "
                         "--parametric-run parametric_exp). Default: for each "
                         "live run, the widest shared run with the same model.")
    ap.add_argument("--csv", type=Path, default=None, help="also write a CSV here")
    args = ap.parse_args()

    available = discover_parametric()
    if not available:
        raise SystemExit(
            f"no shared parametric run under {PARAM_DIR}. Every comparison is "
            "against one, so run it first, e.g.\n"
            "  ./scripts/run_parametric_experiment.py --limit all --run-tag haiku")
    if args.parametric_run and not discover_parametric(args.parametric_run):
        known = ", ".join(f"{t} [{short_model(m)}]" for t, _s, m in available)
        raise SystemExit(f"unknown --parametric-run {args.parametric_run!r}; "
                         f"available: {known}")

    eu = load_eval_utils()
    rows = []
    for system in [s.strip().lower() for s in args.systems.split(",") if s.strip()]:
        if system not in ("tog", "rog"):
            raise SystemExit(f"unknown system {system!r}; choose from ['rog', 'tog']")
        loader = load_tog if system == "tog" else load_rog
        pairs, unmatched, params = build_pairs(system, args.model,
                                               args.allow_model_mismatch,
                                               args.parametric_run)
        print(f"\n=== {system.upper()}")
        if not pairs:
            have = ", ".join(f"{t} [{short_model(m)}]" for t, _s, m in params) or "none"
            print(f"  no live run shares a model with a shared parametric run"
                  f"{f' matching {args.model!r}' if args.model else ''}.")
            print(f"  shared parametric runs available: {have}")
            for tag, model in unmatched:
                print(f"  live {tag} [{short_model(model)}] has no parametric twin")
        for (label, p_tag, p_src, p_model,
             l_tag, l_src, l_model) in pairs:
            param, p_dupes = loader(p_src)
            live, l_dupes = loader(l_src)
            if not param or not live:
                print(f"  [skip] {label}: missing answers "
                      f"({'parametric' if not param else 'live'})")
                continue
            print(f"  {label:<22} {p_tag} [{short_model(p_model)}, "
                  f"{len(param)} q]  vs  {l_tag} [{short_model(l_model)}, "
                  f"{len(live)} q]")
            if p_model != l_model:
                print(f"  [WARN] {label}: model mismatch -- parametric {p_model!r} "
                      f"vs live {l_model!r}. The delta below mixes the model change "
                      f"with the KG change; it is NOT the KG's contribution.")
            for what, dupes in (("parametric", p_dupes), ("live", l_dupes)):
                if dupes:
                    print(f"  [note] {label}: {what} file had {dupes} duplicate id(s) "
                          f"(kept the last answer for each)")
            row = compare(label, param, live, eu)
            if row:
                row["model"] = (p_model if p_model == l_model
                                else f"{p_model} vs {l_model}")
                row["parametric_run"], row["live_run"] = p_tag, l_tag
                rows.append(row)

    if not rows:
        raise SystemExit(
            "nothing to compare. Every comparison is against a shared parametric "
            "run under artifacts/parametric/; run one for the model you want, e.g.\n"
            "  ./scripts/run_parametric_experiment.py --limit all --run-tag haiku")

    width = 113
    print("\n" + "=" * width)
    print(f"{'config':<22}{'nAll':>6}{'pAll':>8}{'pAllF1':>9}"
          f"{'n':>7}{'param':>8}{'live':>8}{'delta':>8}"
          f"{'paramF1':>9}{'liveF1':>8}{'fixed':>7}{'broke':>7}{'net':>6}")
    print("-" * width)
    for r in rows:
        print(f"{r['config']:<22}{r['n_parametric_all']:>6}"
              f"{r['parametric_all_hits1']:>8.2f}{r['parametric_all_f1']:>9.2f}"
              f"{r['n_common']:>7}{r['parametric_hits1']:>8.2f}"
              f"{r['live_hits1']:>8.2f}{r['delta_hits1']:>+8.2f}"
              f"{r['parametric_f1']:>9.2f}{r['live_f1']:>8.2f}"
              f"{r['kg_fixed']:>7}{r['kg_broke']:>7}{r['net']:>+6}")
    print("=" * width)
    print("parametric = the shared run under artifacts/parametric/ (same floor for both systems).")
    print("pAll / pAllF1 = that baseline over ALL nAll of its questions, scored on its own gold;")
    print("  one fixed number per baseline, independent of which live run the row compares.")
    print("param / live / delta = the same two runs over the n questions they SHARE, both")
    print("  scored on the live run's gold -- delta is live minus param, never minus pAll.")
    print("fixed = parametric wrong -> live right;  broke = parametric right -> live wrong.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
