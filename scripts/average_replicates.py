#!/usr/bin/env python3
"""Mean / std-dev across repeat runs of the same config (`<tag>_2`, `<tag>_3`, ...).

A single run of either system is one sample from a sampling LLM: rerunning the
same config moves Hits@1 by a point or two on its own. The `_2` / `_3` tags are
those repeats, and this reports what they actually support -- a mean with a
spread -- rather than a single run's number carried to two decimals.

Two tables per metric:

  ABSOLUTE   each policy's score, averaged over the group's replicates.

Within a run, not across runs: a policy is compared with the `none` from the
same replicate, so the pair differs in the cache policy and nothing else. One
run's semantic_lru against another run's is a backend or model comparison
wearing a policy label.

Replicates are grouped by stripping a trailing `_<digits>` from the tag, and each
group is scored over the INTERSECTION of the question ids its replicates share.
That matters because repeats are rarely the same length (rog_live_virt_gemini_2
holds 391 answers, _3 holds 456): scoring each over its own file would report a
spread that is partly just different question sets. Those counts are per
replicate merges -- a replicate whose policy directory holds several splits
contributes every question any of them answered (scripts/_runs.py), so the
intersection is taken over all the answers on disk rather than over one file
each.

The sweep needs `none` on both sides of the subtraction, so its intersection is
narrower -- the ids every replicate answered under BOTH the policy and `none` --
and it gets its own `shared` column instead of being folded into the absolute
one. The gap can be large: gemini_tog_cache_virtuoso's policies hold 1639
answers each but its `none` holds 149, so the absolute rows stand on 1639
questions and the sweep rows on 149. Renumbering the absolute table down to 149
would throw away most of what those runs measured; hiding the 149 would make the
deltas look better founded than they are.

Per policy, not one intersection across the whole sweep, for the reason
every other policy's row.

The unsuffixed base tag is NOT included by default. `rog_live_virt_gemini` exists
alongside `_2`/`_3` but was run over the full split rather than the 400-question
one, so folding it in would average two different configurations. `--include-base`
opts in where you know they match. `--first N` is one place they do: cut to a
prefix both runs were asked, the split length stops being a difference between
them, and `--include-base --first 300` gives those groups a third replicate.

The original is not always the unsuffixed tag. ToG writes its as
`tog_rerun_live_virt_gemini` for the `tog_live_virt_gemini_2`/`_3` pair -- same
identity in run_config.json (webqsp, tamu, gemini-3.1-flash-lite, depth 3,
width 3, t=0.90, capacity 4096) apart from `test_limit`, which is what `--first`
answers -- so `--include-base` looks for the `tog_rerun_` spelling too, and
`--base-tag GROUP=TAG` names it where a group is spelled some third way.
`gemini_tog_cache_virtuoso` and `gemini_rog_cache_virtuoso` have no original on
disk under either spelling and stay at n=2.

ToG answers use the last-braced-span rule (scripts/_runs.py), the same one
rescore_tog.py applies -- eval.py's first-span scan returns the `{Yes}` marker.

`--universe SPLIT` replaces that intersection with a fixed denominator: the ids
stage 1 was asked for SPLIT, with a question stage 2 never answered scored as a
miss. That is a claim about WHY records go missing, and for these runs it holds
-- stage 1 answered 400/400 in both replicates, stage 2 wrote 391, and the gap
is predict_answer_api.py catching an exception from the reasoner's LLM call and
returning None. Those are questions the system failed to answer, not questions
it was never asked, and the gaps are scattered through the split rather than a
truncated tail. Use it only where that is true: against a run cut short, or one
whose failures are transient API trouble rather than the system's own, it prices
infrastructure as if it were accuracy.

    ./scripts/average_replicates.py
    ./scripts/average_replicates.py --system rog --group rog_live_virt_gemini
    ./scripts/average_replicates.py --no-sweep          # absolute tables only
    ./scripts/average_replicates.py --universe 'test[:400]'   # misses score 0
    ./scripts/average_replicates.py --first 300   # only the first 300 asked
    ./scripts/average_replicates.py --csv artifacts/replicate_variance.csv

`--first N` cuts every id set down to the first N questions the group was
asked, in the order it asked them, and changes nothing else: the intersections,
the universe and the scoring are what they were, taken over a smaller pool. The
order comes from the longest single answer file in the group (scripts/_runs.py
`ask_order`), and the other files are checked to run in that same order before
it is used, so a group whose replicates were asked in different orders is left
whole rather than cut at a boundary that means something different in each.

With n=2 the std-dev is just |a-b|/sqrt(2); it is printed, and flagged, because
two runs bound the spread loosely rather than estimating it.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runs import (COMPARE_DIR, ROG_CACHE_DIR, ask_order,  # noqa: E402
                   is_subsequence, load_answers, load_eval_utils, rog_universe,
                   score)

REPLICATE_RE = re.compile(r"^(?P<base>.+?)_(?P<rep>\d+)$")
METRICS = ("hits1", "f1", "precision", "recall")
# RoG's eval_result.txt prints Accuracy and Recall as the same number, and
# summarize_tog_cache.py sets accuracy = recall, so name it that way here too.
METRIC_LABELS = {"hits1": "Hits@1", "recall": "Accuracy (recall)",
                 "f1": "F1", "precision": "Precision"}
# The uncached policy every other one is swept against.
BASE_POLICY = "none"


def base_candidates(system: str, base: str) -> list[str]:
    """Tags that could hold the group's original run, best guess first.

    ToG spells its originals with `rerun_` in the middle -- the run
    `tog_live_virt_gemini_2` and `_3` repeat is on disk as
    `tog_rerun_live_virt_gemini` -- so the plain name is tried first and that
    spelling after it. Both are checked against `run_config.json` by the caller
    only in the sense that they are the same identity apart from `test_limit`;
    `--base-tag` overrides this where a group is named some third way.
    """
    cands = [base]
    if system == "tog" and base.startswith("tog_"):
        cands.append("tog_rerun_" + base[len("tog_"):])
    return cands


def group_runs(system: str, include_base: bool,
               base_tags: dict[str, str] | None = None) -> tuple[dict, dict]:
    """({base tag: [replicate tags]}, {base tag: original run folded in})."""
    root = COMPARE_DIR if system == "tog" else ROG_CACHE_DIR
    if not root.is_dir():
        return {}, {}
    groups: dict[str, list[str]] = defaultdict(list)
    names = {d.name for d in root.iterdir() if d.is_dir()}
    for name in sorted(names):
        m = REPLICATE_RE.match(name)
        if not m:
            continue
        base = m.group("base")
        # `_128` / `_512` are cache capacities, not replicate numbers: a run
        # named for its capacity is a different config, not a repeat of one.
        if len(m.group("rep")) > 1:
            continue
        groups[base].append(name)
    used: dict[str, str] = {}
    if include_base:
        for base in list(groups):
            explicit = (base_tags or {}).get(base)
            cands = [explicit] if explicit else base_candidates(system, base)
            for cand in cands:
                if cand in names:
                    groups[base].insert(0, cand)
                    used[base] = cand
                    break
            else:
                # Reported by the caller, so the note lands under the system
                # header rather than ahead of it.
                used[base] = ""
    return {b: v for b, v in groups.items() if len(v) >= 2}, used


def policies_of(system: str, tag: str) -> set[str]:
    """Policy names a run holds answers for."""
    if system == "tog":
        d = COMPARE_DIR / tag
        return {f.name[: -len(".jsonl")] for f in d.glob("*.jsonl")
                if not f.name.endswith(".metrics.jsonl")} if d.is_dir() else set()
    kgqa = ROG_CACHE_DIR / tag / "KGQA"
    if not kgqa.is_dir():
        return set()
    return {p.name for p in kgqa.iterdir()
            if p.is_dir() and any(p.glob("**/predictions.jsonl"))}


def load_replicates(system: str, tags: list[str], policy: str) -> list | None:
    """[(tag, {id: record})] for one policy; None if any replicate is missing it."""
    loaded = []
    for tag in tags:
        _path, recs, dupes = load_answers(system, tag, policy)
        if not recs:
            return None
        if dupes:
            print(f"  [note] {tag}:{policy}: {dupes} duplicate id(s), kept the last")
        loaded.append((tag, recs))
    return loaded


def summarise(system: str, base: str, loaded: list, policy: str,
              none_loaded: list | None, eu, universe: set | None = None,
              first: set | None = None) -> dict | None:
    """Mean/std across replicates for one policy, plus its sweep against `none`.

    Without a universe the absolute figures are scored on the ids the replicates
    share, and the sweep on the narrower set they share under both this policy
    and `none`. With one, both are scored on that fixed id set and a question a
    replicate never answered counts as a miss (see `--universe`).

    `first` is `--first`'s id set, and intersects whichever of those it is: the
    pool shrinks, nothing else about the scoring moves.
    """
    if universe is not None:
        common = sorted(universe)
        coverage = "/".join(str(len(universe & set(recs))) for _t, recs in loaded)
    else:
        common = set(loaded[0][1])
        for _tag, recs in loaded[1:]:
            common &= set(recs)
        if first is not None:
            common &= first
        if not common:
            print(f"  [skip] {base}:{policy}: replicates share no question ids"
                  + (" among the first ones asked" if first is not None else ""))
            return None
        common = sorted(common)
        coverage = ""

    per_run = {tag: score(recs, common, eu) for tag, recs in loaded}
    row = {"system": system, "group": base, "policy": policy,
           "n_replicates": len(loaded), "n_shared": len(common),
           "first_n": len(first) if first is not None else "",
           "answered": coverage,
           "file_sizes": "/".join(str(len(r)) for _t, r in loaded)}
    for metric in METRICS:
        vals = [per_run[tag][metric] for tag, _ in loaded]
        row[f"{metric}_mean"] = round(statistics.mean(vals), 2)
        row[f"{metric}_std"] = round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0
        row[f"{metric}_min"] = round(min(vals), 2)
        row[f"{metric}_max"] = round(max(vals), 2)
        row[f"{metric}_each"] = [round(v, 2) for v in vals]
    row["runs"] = " ".join(tag for tag, _ in loaded)

    # Sweep columns exist on every row so the CSV keeps one header; they stay
    # empty for `none` itself (it is the baseline) and where it is missing.
    row["n_shared_vs_none"] = ""
    row["net_mean"] = ""
    row["net_each"] = ""
    for metric in METRICS:
        for suffix in ("base_mean", "delta_mean", "delta_std",
                       "delta_min", "delta_max", "delta_each"):
            row[f"{metric}_{suffix}"] = ""
    if policy != BASE_POLICY and none_loaded:
        _add_sweep(row, base, policy, loaded, dict(none_loaded), eu, universe,
                   first)
    return row


def _add_sweep(row: dict, base: str, policy: str, loaded: list,
               none_by_tag: dict, eu, universe: set | None = None,
               first: set | None = None) -> None:
    """Fill in policy-minus-`none`, replicate by replicate, on their shared ids."""
    if any(tag not in none_by_tag for tag, _ in loaded):
        return
    if universe is not None:
        # Both sides already answer to the same fixed denominator, so the sweep
        # needs no narrower intersection -- and a policy that crashes on a
        # question no longer scores as if it had never been asked.
        vs = sorted(universe)
    else:
        vs = set(loaded[0][1]) & set(none_by_tag[loaded[0][0]])
        for tag, recs in loaded[1:]:
            vs &= set(recs) & set(none_by_tag[tag])
        if first is not None:
            vs &= first
        if not vs:
            print(f"  [skip] {base}:{policy}: shares no question ids with `none`")
            return
        vs = sorted(vs)

    pol = {tag: score(recs, vs, eu) for tag, recs in loaded}
    unc = {tag: score(none_by_tag[tag], vs, eu) for tag, _ in loaded}
    row["n_shared_vs_none"] = len(vs)
    for metric in METRICS:
        deltas = [pol[tag][metric] - unc[tag][metric] for tag, _ in loaded]
        row[f"{metric}_base_mean"] = round(
            statistics.mean([unc[tag][metric] for tag, _ in loaded]), 2)
        row[f"{metric}_delta_mean"] = round(statistics.mean(deltas), 2)
        row[f"{metric}_delta_std"] = (round(statistics.stdev(deltas), 2)
                                      if len(deltas) > 1 else 0.0)
        row[f"{metric}_delta_min"] = round(min(deltas), 2)
        row[f"{metric}_delta_max"] = round(max(deltas), 2)
        row[f"{metric}_delta_each"] = [round(d, 2) for d in deltas]
    # Questions the policy fixed minus questions it broke, per replicate. Hits
    # based, so it belongs to Hits@1: a delta of zero can be no change at all or
    # equal numbers of both, and only this separates them.
    nets = []
    for tag, _recs in loaded:
        p, u = pol[tag]["per_question"], unc[tag]["per_question"]
        nets.append(sum(1 for q in vs if p[q] and not u[q])
                    - sum(1 for q in vs if u[q] and not p[q]))
    row["net_mean"] = round(statistics.mean(nets), 1)
    row["net_each"] = nets


def _universe_for(system: str, base: str, tags: list[str],
                  split: str) -> set | None:
    """The fixed id set for `--universe`, or None to fall back to intersections.

    Every replicate must have been asked the same questions for the denominator
    to mean anything, so a group whose stage-1 files disagree is reported and
    left on the intersection rather than scored against a set some replicate was
    never asked.
    """
    if not split:
        return None
    if system != "rog":
        print(f"  [note] {base}: --universe needs stage-1 predictions, which "
              "only RoG runs write; scoring on the intersection instead")
        return None
    per_tag = {tag: rog_universe(tag, BASE_POLICY, split) for tag in tags}
    missing = [tag for tag, ids in per_tag.items() if not ids]
    if missing:
        print(f"  [note] {base}: no stage-1 {split!r} for {', '.join(missing)}; "
              "scoring on the intersection instead")
        return None
    sizes = {tag: len(ids) for tag, ids in per_tag.items()}
    first = per_tag[tags[0]]
    if any(ids != first for ids in per_tag.values()):
        print(f"  [note] {base}: replicates were asked different {split!r} "
              f"questions ({sizes}); scoring on the intersection instead")
        return None
    print(f"  [universe] {base}: {len(first)} questions from stage-1 {split!r}; "
          "unanswered ones score zero")
    return first


def _first_ids(system: str, base: str, tags: list[str], policies, n: int) -> set | None:
    """The first `n` question ids this group was asked, or None to score them all.

    One order per group, not one per run: the policies are being compared with
    each other, so cutting each at its own 300 would hand them different
    question sets and call the difference a policy effect. The longest file's
    order is the reference, and every other file must run in that same order --
    a subsequence of it, since a shorter run answers fewer questions rather than
    different ones. Where that does not hold the group keeps all its questions
    and says so, because there is then no single "first 300" to speak of.
    """
    if n <= 0:
        return None
    orders = {}
    for tag in tags:
        for policy in sorted(set(policies) | {BASE_POLICY}):
            order = ask_order(system, tag, policy)
            if order:
                orders[f"{tag}:{policy}"] = order
    if not orders:
        print(f"  [note] {base}: no file records the order questions were asked "
              "in; scoring all of them")
        return None
    ref_name, ref = max(orders.items(), key=lambda kv: len(kv[1]))
    disagree = [name for name, order in orders.items()
                if not is_subsequence(order, ref)]
    if disagree:
        print(f"  [note] {base}: {', '.join(disagree)} were not asked in the "
              f"same order as {ref_name}; scoring all questions instead of the "
              f"first {n}")
        return None
    if len(ref) <= n:
        print(f"  [first] {base}: asked {len(ref)} questions, at or under the "
              f"first {n}; nothing dropped")
        return set(ref)
    print(f"  [first] {base}: the first {n} of {len(ref)} questions asked "
          f"(order from {ref_name})")
    return set(ref[:n])


def _absolute_table(rows: list[dict], metric: str, title: str) -> None:
    # `shared` carries the per-replicate answered counts under --universe, so it
    # is sized for "400 (391/391)" rather than for a bare id count.
    scored = "scored" if any(r["answered"] for r in rows) else "shared"
    width = 115 if scored == "scored" else 106
    print(f"\n{title} variation across replicates")
    print("=" * width)
    print(f"{'system':<7}{'group':<28}{'policy':<22}{'n':>3}{scored:>17}"
          f"{'mean':>9}{'std':>8}{'min':>8}{'max':>8}{'range':>8}")
    print("-" * width)
    for r in rows:
        lo, hi = r[f"{metric}_min"], r[f"{metric}_max"]
        shared = (f"{r['n_shared']} ({r['answered']})" if r["answered"]
                  else str(r["n_shared"]))
        print(f"{r['system']:<7}{r['group']:<28}{r['policy']:<22}"
              f"{r['n_replicates']:>3}{shared:>17}"
              f"{r[f'{metric}_mean']:>9.2f}{r[f'{metric}_std']:>8.2f}"
              f"{lo:>8.2f}{hi:>8.2f}{hi - lo:>8.2f}")
    print("=" * width)
    print("  per replicate: " + " | ".join(
        f"{r['group']}:{r['policy']} " + " ".join(f"{v:.2f}" for v in r[f"{metric}_each"])
        for r in rows))


def _sweep_table(rows: list[dict], metric: str, title: str) -> None:
    """One row per cached policy: its mean gain over the uncached run, +/- spread."""
    swept = [r for r in rows if r["n_shared_vs_none"] != ""]
    if not swept:
        return
    width = 114 if metric == "hits1" else 106
    print(f"\n{title} vs the same run's uncached `none`, averaged over replicates")
    print("=" * width)
    header = (f"{'system':<7}{'group':<28}{'policy':<22}{'n':>3}{'shared':>8}"
              f"{'none':>8}{'policy':>8}{'delta':>9}{'std':>7}{'worst':>8}")
    print(header + (f"{'net':>8}" if metric == "hits1" else ""))
    print("-" * width)
    for r in swept:
        base = r[f"{metric}_base_mean"]
        delta = r[f"{metric}_delta_mean"]
        print(f"{r['system']:<7}{r['group']:<28}{r['policy']:<22}"
              f"{r['n_replicates']:>3}{r['n_shared_vs_none']:>8}"
              f"{base:>8.2f}{base + delta:>8.2f}{delta:>+9.2f}"
              f"{r[f'{metric}_delta_std']:>7.2f}{r[f'{metric}_delta_min']:>+8.2f}"
              + (f"{r['net_mean']:>+8.1f}" if metric == "hits1" else ""))
    print("=" * width)
    print("  per replicate: " + " | ".join(
        f"{r['group']}:{r['policy']} "
        + " ".join(f"{v:+.2f}" for v in r[f"{metric}_delta_each"])
        for r in swept))
    print("  'shared' is this policy's overlap with `none`, NOT the absolute "
          "table's; 'worst' is the")
    print("  weakest single replicate's delta.")
    if metric == "hits1":
        print("  net = questions the cache fixed minus questions it broke, "
              "averaged over replicates.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="tog,rog",
                    help="comma-separated: tog, rog (default: both)")
    ap.add_argument("--group", nargs="+", default=None,
                    help="only these base tags (default: every group found)")
    ap.add_argument("--policy", default="",
                    help="only this policy (default: every policy all "
                         "replicates share)")
    ap.add_argument("--metrics", default="hits1,recall,f1",
                    help="one table pair per metric, in order. Choose from "
                         f"{', '.join(METRICS)} (default: hits1,recall,f1 -- "
                         "recall is what this project calls accuracy, and F1 "
                         "pairs it with precision so a policy that answers "
                         "with longer lists cannot buy recall for free)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="absolute tables only; skip the policy-vs-`none` sweep")
    ap.add_argument("--universe", default="",
                    help="RoG only: score every replicate over the ids stage 1 "
                         "was asked for this SPLIT (e.g. 'test[:400]') instead "
                         "of over the ids the replicates share, counting a "
                         "question stage 2 never answered as a miss. Fixes the "
                         "denominator and folds reasoner failures into the "
                         "score; see the note at the top of this file.")
    ap.add_argument("--first", type=int, default=0, metavar="N",
                    help="score only the first N questions the group was asked, "
                         "in the order it asked them (default: 0, all of them). "
                         "Shrinks the pool every id set is taken from; the "
                         "intersections and the scoring are otherwise unchanged.")
    ap.add_argument("--include-base", action="store_true",
                    help="also treat the group's original run as a replicate "
                         "(the unsuffixed tag, or ToG's `tog_rerun_*` spelling "
                         "of it). Only correct when it really is the same "
                         "configuration -- with `--first` cutting both to a "
                         "prefix they were each asked, a differing `test_limit` "
                         "no longer counts as a difference.")
    ap.add_argument("--base-tag", action="append", default=[], metavar="GROUP=TAG",
                    help="the original run for GROUP is on disk as TAG; "
                         "repeatable. Only used with --include-base, and only "
                         "needed where the tag is spelled some third way.")
    ap.add_argument("--csv", type=Path, default=None, help="also write a CSV here")
    args = ap.parse_args()
    base_tags = {}
    for pair in args.base_tag:
        if "=" not in pair:
            raise SystemExit(f"--base-tag wants GROUP=TAG, got {pair!r}")
        group, tag = pair.split("=", 1)
        base_tags[group] = tag

    eu = load_eval_utils()
    rows = []
    for system in [s.strip().lower() for s in args.systems.split(",") if s.strip()]:
        if system not in ("tog", "rog"):
            raise SystemExit(f"unknown system {system!r}")
        groups, base_used = group_runs(system, args.include_base, base_tags)
        if args.group:
            groups = {b: v for b, v in groups.items() if b in set(args.group)}
        print(f"\n=== {system.upper()}")
        if not groups:
            print("  no replicate groups found")
            continue
        for base, tags in sorted(groups.items()):
            shared_policies = set.intersection(
                *(policies_of(system, t) for t in tags)) or set()
            if args.policy:
                shared_policies &= {args.policy}
            if not shared_policies:
                print(f"  [skip] {base}: replicates {tags} share no policy")
                continue
            print(f"  {base}: {len(tags)} replicates {tags}")
            if base in base_used and not base_used[base]:
                print(f"  [note] {base}: no original run on disk under "
                      f"{' or '.join(base_candidates(system, base))}; "
                      "replicates only")
            elif base_used.get(base, base) != base:
                print(f"  [base] {base}: folded in the original run "
                      f"{base_used[base]}")
            universe = _universe_for(system, base, tags, args.universe)
            first_ids = _first_ids(system, base, tags, shared_policies,
                                   args.first)
            if universe is not None and first_ids is not None:
                # Restricted here rather than in `summarise` so the fixed
                # denominator and its answered counts shrink together.
                universe = universe & first_ids
            # Loaded once per group: every policy in the sweep subtracts it.
            none_loaded = (None if args.no_sweep
                           else load_replicates(system, tags, BASE_POLICY))
            if none_loaded is None and not args.no_sweep:
                print(f"  [note] {base}: no `none` on every replicate, "
                      "reporting absolute figures only")
            for policy in sorted(shared_policies):
                loaded = load_replicates(system, tags, policy)
                if loaded is None:
                    continue
                row = summarise(system, base, loaded, policy, none_loaded, eu,
                                universe, first_ids)
                if row:
                    rows.append(row)

    if not rows:
        raise SystemExit("no replicate groups to average")

    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        if f"{metric}_mean" not in rows[0]:
            raise SystemExit(f"unknown metric {metric!r}; "
                             f"choose from {', '.join(METRICS)}")
        title = METRIC_LABELS.get(metric, metric)
        _absolute_table(rows, metric, title)
        if not args.no_sweep:
            _sweep_table(rows, metric, title)

    if any(r["answered"] for r in rows):
        print("\nA `scored` cell reading `400 (391/391)` is 400 questions in the "
              "denominator, of which each replicate answered 391: the 9 the "
              "reasoner failed on score zero rather than leaving the run.")
    if args.first > 0:
        print(f"\nRestricted to the first {args.first} questions each group was "
              "asked: every id set below is its usual one intersected with "
              "those, so a group that was asked fewer is unaffected.")
    print("\nAbsolute metrics are over the questions all replicates in a group "
          "answered, except where `scored` says otherwise (file sizes: "
          + "; ".join(f"{r['group']}:{r['policy']} {r['file_sizes']}" for r in rows)
          + ").")
    print("Accuracy is recall of the gold answers -- the same quantity RoG's "
          "eval_result.txt prints as `Accuracy`.")
    if any(r["n_replicates"] == 2 for r in rows):
        print("NOTE: with n=2 the std-dev is |a-b|/sqrt(2) -- it bounds the spread, "
              "it does not estimate it.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
