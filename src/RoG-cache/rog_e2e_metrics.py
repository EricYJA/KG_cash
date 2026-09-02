"""End-to-end (stage 1 + stage 2) per-question timing for a RoG cache run.

ToG times a whole question inside one loop, so its `full_speedup_x` really is a
full-system number. RoG splits the same work across two *processes* -- stage 1
(planner, cached) and stage 2 (reasoner, never cached) -- and each only ever
timed itself. Timing stage 1 alone made `full_speedup_x` a **planner-stage**
speedup that silently excluded every second of stage 2, which is where the
reasoner's LLM call and the path grounding live. A cache that halves stage 1
cannot halve a pipeline whose second half it never touches, so the stage-1
number is an upper bound on the real one, not the real one.

This module joins the two stages' per-question sidecars on question id, sums
their elapsed times into one whole-question record, and hands those records to
ToG's own `aggregate_run_metrics()`. Same record shape, same function, same
formula -- so a RoG `full_speedup_x` and a ToG `full_speedup_x` finally mean the
same thing and can share a plot axis.

The join is restart-safe for the same reason the sidecars are: a resumed run
appends new records, and the last record for an id wins.
"""
from __future__ import annotations

import json
import os
import sys

# ToG's metrics helpers are the reference implementation -- reused rather than
# reimplemented so the two systems' numbers come out of the *same* function.
# In-container /togcache is on PYTHONPATH; on the host, fall back to the path.
try:
    from cache_metrics import (
        aggregate_run_metrics,
        append_question_metrics,
        metrics_sidecar_path,
    )
except ImportError:  # pragma: no cover - host-side convenience
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ToG-cache",
            "ToG",
        ),
    )
    from cache_metrics import (
        aggregate_run_metrics,
        append_question_metrics,
        metrics_sidecar_path,
    )

# Re-exported so both RoG stages import their sidecar helpers from one place
# and cannot drift onto a second, RoG-local copy of ToG's record format.
__all__ = [
    "aggregate_run_metrics",
    "append_question_metrics",
    "metrics_sidecar_path",
    "load_sidecar",
    "merge_stage_records",
    "aggregate_end_to_end",
]


def load_sidecar(path):
    """{question id: record} from a per-question sidecar JSONL.

    Later records win: a question that was retried after a crash (or after a
    failed LLM call, which upstream leaves out of predictions.jsonl so the
    resume redoes it) appears more than once, and the last attempt is the one
    that produced the output being scored.
    """
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("id")
            if qid is not None:
                out[qid] = record
    return out


def merge_stage_records(stage1_path, stage2_path):
    """Join the two stages into whole-question records, ToG's record shape.

    Returns (records, dropped). A question is only counted when *both* stages
    timed it: stage 2 drops questions whose reasoner call failed (and
    --filter_empty drops more), and half a pipeline is not an end-to-end time.
    Those ids are reported as `dropped` rather than silently folded in at their
    stage-1 time, which would understate the miss cost and inflate the speedup.
    """
    stage1 = load_sidecar(stage1_path)
    stage2 = load_sidecar(stage2_path)

    records, dropped = [], []
    for qid, s1 in stage1.items():
        s2 = stage2.get(qid)
        if s2 is None:
            dropped.append(qid)
            continue
        s1_elapsed = float(s1.get("elapsed_s", 0.0) or 0.0)
        s2_elapsed = float(s2.get("elapsed_s", 0.0) or 0.0)
        records.append(
            {
                "id": qid,
                "question": s1.get("question"),
                # The cache only exists in stage 1, so it defines hit/miss for
                # the whole question -- exactly as a ToG chain-cache hit labels
                # the whole ToG question.
                "cache_hit": bool(s1.get("cache_hit")),
                "cache_hit_type": s1.get("cache_hit_type"),
                # Either stage dying in the client's retry loop makes the whole
                # question's elapsed time a measurement of the vendor being down.
                # Propagated so the aggregator still counts it as a question but
                # keeps it out of the averages the speedups are priced against;
                # a sidecar written before the flag existed carries no `failed`
                # key and comes through as it always did.
                "failed": bool(s1.get("failed") or s2.get("failed")),
                "elapsed_s": s1_elapsed + s2_elapsed,
                "llm_calls": int(s1.get("llm_calls", 0) or 0)
                + int(s2.get("llm_calls", 0) or 0),
                # Kept so the stage split stays inspectable after the join --
                # this is what shows how much of a question the cache can reach.
                "stage1_s": round(s1_elapsed, 3),
                "stage2_s": round(s2_elapsed, 3),
            }
        )
    return records, dropped


def aggregate_end_to_end(stage1_path, stage2_path, merged_path=None):
    """(timing, summary, breakdown, dropped) over whole-question times.

    `merged_path` (optional) receives the joined records as their own sidecar,
    which is what makes this "exactly like ToG": the aggregation below is ToG's
    `aggregate_run_metrics` reading a ToG-shaped per-question JSONL, not a
    reimplementation of its arithmetic here.
    """
    records, dropped = merge_stage_records(stage1_path, stage2_path)
    if not records:
        return {}, {}, {}, dropped

    path = merged_path
    if path is None:
        # aggregate_run_metrics reads a file, so a run without a merged-sidecar
        # destination still needs one; put it beside stage 1's.
        path = f"{stage1_path}.e2e.jsonl"
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:  # rebuilt from scratch: the join is idempotent
        for record in records:
            f.write(json.dumps(record) + "\n")

    timing, summary, breakdown, _per_loop = aggregate_run_metrics(path)
    summary = dict(summary)
    summary["e2e_metrics_file"] = path
    summary["questions_dropped_no_stage2"] = len(dropped)
    return timing, summary, breakdown, dropped
