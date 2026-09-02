#!/usr/bin/env python3
"""Time-saved cache simulation for RoG -- the analogue of ToG's cache_simulator.py.

ToG replays a captured stream of *KG requests* and asks how much SPARQL time a
cache would remove. RoG issues no KG requests at all (it grounds paths against
the subgraph shipped inside each dataset record), so that simulator has nothing
to replay. What RoG does have is a per-question planner LLM call, and that is
what its cache elides.

So the replayable trace here is the per-question timing sidecar of an
*uncached* run: `artifacts/rog_cache/<run>/.../none/.../*.metrics.jsonl`,
written by gen_rule_path_api.py (stage 1) and predict_answer_api.py (stage 2).
Those files record, for every question, what the planner and the reasoner
actually cost with no cache in the way. A cache policy then only decides which
of those measured planner costs you still pay:

    planner_simulated = sum(stage1_s over misses) + hits * hit_cost
    reasoner          = sum(stage2_s)              # never cached, always paid
    saved             = planner_base - planner_simulated

This is the same accounting ToG uses for `kg_base` / `kg_simulated` / `saved`,
so the output JSON mirrors cache_sim_summary.json field for field and the same
plotting and table code reads both. Unlike simulate_rog_cache.py (hit rate
only) this yields the end-to-end speedup columns; unlike the real pipeline
(scripts/run_rog_cache_experiment.py) it needs no LLM, no GPU and no re-run,
because every number it multiplies was already measured once.

What it does NOT tell you is accuracy. A semantic hit transplants another
question's relation paths, and whether those still ground to the right answer
is only answerable by actually running stage 2. Use the live experiment for that.

Usage:
    python simulate_rog_cache_timing.py --run artifacts/rog_cache/rog_cache_virtuoso_new
    python simulate_rog_cache_timing.py --run <dir> --cache-sizes 10,50,100,500,1000 \
        --policies exact,semantic_lru,semantic_lfu,semantic_fifo,semantic_random,semantic_belady,semantic_oracle
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import sys
from pathlib import Path

import datasets_compat  # noqa: F401  (read caches written by a newer datasets)

_TOG_DIR = os.environ.get("TOG_CACHE_DIR")
if not _TOG_DIR or not os.path.isdir(_TOG_DIR):
    _candidate = Path(__file__).resolve().parents[1] / "ToG-cache" / "ToG"
    _TOG_DIR = str(_candidate) if _candidate.is_dir() else "/togcache"
sys.path.insert(0, _TOG_DIR)

from simulate_cache import (  # noqa: E402  (ToG's simulator internals, reused as-is)
    FastSimCache,
    SemanticBeladyCache,
    precompute_embeddings,
)

DEFAULT_POLICIES = (
    "exact",
    "semantic_lru",
    "semantic_lfu",
    "semantic_fifo",
    "semantic_random",
    "semantic_belady",
    "semantic_oracle",
)
DEFAULT_CACHE_SIZES = (10, 50, 100, 500, 1000)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "rog_cache_sim" / "rog_cache_sim_summary.json"
)


# --------------------------------------------------------------------------
# Loading the uncached trace
# --------------------------------------------------------------------------


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _find_one(run_dir: Path, pattern: str, what: str) -> str:
    matches = sorted(glob.glob(str(run_dir / pattern), recursive=True))
    if not matches:
        raise SystemExit(
            f"no {what} sidecar under {run_dir} matching {pattern!r}.\n"
            f"This simulator replays an *uncached* run: point --run at a directory "
            f"that contains a completed `none` policy with its .metrics.jsonl files."
        )
    if len(matches) > 1:
        print(f"[warn] {len(matches)} {what} sidecars matched; using {matches[0]}")
    return matches[0]


def load_uncached_trace(run_dir: Path, policy_tag: str = "none") -> tuple[list[dict], dict]:
    """Join the stage-1 and stage-2 sidecars of an uncached run.

    Returns (records, provenance) where each record is
    {id, question, stage1_ms, stage2_ms} in the order stage 1 processed them.
    """
    stage1_path = _find_one(
        run_dir, f"gen_rule_path/{policy_tag}/**/*.metrics.jsonl", "stage-1 (planner)"
    )
    stage2_path = _find_one(
        run_dir, f"KGQA/{policy_tag}/**/predictions.jsonl.metrics.jsonl", "stage-2 (reasoner)"
    )

    # Later records win: a resumed run re-appends questions it retried.
    stage1 = {r["id"]: r for r in _read_jsonl(stage1_path)}
    stage2 = {r["id"]: r for r in _read_jsonl(stage2_path)}

    order = [r["id"] for r in _read_jsonl(stage1_path)]
    seen: set = set()
    ordered_ids = [i for i in order if not (i in seen or seen.add(i))]

    records = []
    dropped = 0
    for qid in ordered_ids:
        s1 = stage1[qid]
        s2 = stage2.get(qid)
        if s2 is None:
            dropped += 1
            continue
        if s1.get("cache_hit"):
            # Would understate the miss cost: this question's planner call was
            # served from a warm cache, so its elapsed_s is a hit, not the
            # uncached cost this simulator needs.
            dropped += 1
            continue
        records.append(
            {
                "id": qid,
                "question": s1["question"],
                "stage1_ms": int(round(float(s1["elapsed_s"]) * 1000)),
                "stage2_ms": int(round(float(s2["elapsed_s"]) * 1000)),
            }
        )

    if not records:
        raise SystemExit(f"no usable uncached questions found in {run_dir}")

    provenance = {
        "run_dir": str(run_dir),
        "policy_tag": policy_tag,
        "stage1_metrics": stage1_path,
        "stage2_metrics": stage2_path,
        "questions_dropped": dropped,
    }
    return records, provenance


def load_questions_from_dataset(data_path: str, dataset: str, split: str, limit) -> tuple[list[dict], dict]:
    """Question sequence straight from the dataset, with no measured timings.

    Used for a dataset that has no uncached run to replay (RoG-cwq has none).
    Hit rate only depends on the question sequence, so the hit-rate figures are
    exactly as valid as they are for a dataset with sidecars; the time columns
    are simply zero, and metadata records that they are unavailable.
    """
    from datasets import load_dataset

    data = load_dataset(os.path.join(data_path, dataset), split=split)
    if limit is not None:
        data = data.select(range(min(limit, len(data))))
    records = [
        {"id": row["id"], "question": row["question"], "stage1_ms": 0, "stage2_ms": 0}
        for row in data
    ]
    return records, {
        "source": f"{data_path}/{dataset}[{split}]",
        "timing_available": False,
        "questions_dropped": 0,
    }


def load_oracle_keys(records: list[dict], data_path: str, dataset: str, split: str) -> dict:
    """Gold-answer keys per question id, for the semantic_oracle policy."""
    try:
        from datasets import load_dataset

        from rog_question_cache import extract_oracle_answer_key
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot load gold answers ({exc}); semantic_oracle will be all-miss")
        return {}
    try:
        data = load_dataset(os.path.join(data_path, dataset), split=split)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] cannot load {dataset} ({exc}); semantic_oracle will be all-miss")
        return {}
    keys = {}
    for row in data:
        keys[row["id"]] = extract_oracle_answer_key(row, dataset)
    missing = sum(1 for r in records if not keys.get(r["id"]))
    if missing:
        print(f"[info] {missing}/{len(records)} questions have no extractable gold-answer key")
    return keys


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def simulate_one(
    records: list[dict],
    oracle_keys: dict,
    policy: str,
    cache_size: int,
    threshold: float,
    embedder_model: str,
    embed_map: dict,
    hit_cost_ms: int,
    random_seed: int,
) -> dict:
    """Replay `records` under one (policy, cache_size) and return the result dict."""
    questions = [r["question"] for r in records]
    if policy == "semantic_belady":
        cache = SemanticBeladyCache(
            path="",
            capacity=cache_size,
            policy=policy,
            similarity_threshold=threshold,
            embedder_model=embedder_model,
            embed_map=embed_map,
            question_positions=questions,
        )
    else:
        cache = FastSimCache(
            path="",
            capacity=cache_size,
            policy=policy,
            similarity_threshold=threshold,
            embedder_model=embedder_model,
            embed_map=embed_map,
            random_seed=random_seed,
        )

    planner_base_ms = 0
    planner_simulated_ms = 0
    reasoner_ms = 0
    hits = 0
    for index, record in enumerate(records):
        if isinstance(cache, SemanticBeladyCache):
            cache.set_position(index)
        planner_base_ms += record["stage1_ms"]
        reasoner_ms += record["stage2_ms"]
        oracle_key = oracle_keys.get(record["id"]) if policy == "semantic_oracle" else None
        if cache.get(record["question"], oracle_key=oracle_key) is not None:
            hits += 1
            planner_simulated_ms += hit_cost_ms
            continue
        planner_simulated_ms += record["stage1_ms"]
        cache.put(record["question"], ["DUMMY_RELATION_PATH"], oracle_key=oracle_key)

    misses = len(records) - hits
    total_base = planner_base_ms + reasoner_ms
    total_simulated = planner_simulated_ms + reasoner_ms
    return {
        "policy": policy,
        "cache_size": cache_size,
        "requests": len(records),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / len(records), 4) if records else 0.0,
        "time_breakdown_ms": {
            "planner_base": planner_base_ms,
            "planner_simulated": planner_simulated_ms,
            "reasoner": reasoner_ms,
            "other": 0,
            "total_base": total_base,
            "total_simulated": total_simulated,
            "saved": total_base - total_simulated,
        },
    }


def build_dataset_payload(
    records: list[dict],
    provenance: dict,
    oracle_keys: dict,
    cache_sizes: list[int],
    policies: list[str],
    threshold: float,
    embedder_model: str,
    hit_cost_ms: int,
    shuffle_seed: int,
    random_seed: int,
) -> dict:
    embed_map = precompute_embeddings(
        [{"question": r["question"]} for r in records], "question", embedder_model
    )

    shuffled = copy.copy(records)
    random.Random(shuffle_seed).shuffle(shuffled)

    orderings = {"sequential": records, "shuffled": shuffled}
    dataset_payload: dict = {
        **provenance,
        "request_count": len(records),
        "time_breakdown_ms": {
            "planner": sum(r["stage1_ms"] for r in records),
            "reasoner": sum(r["stage2_ms"] for r in records),
            "other": 0,
            "total": sum(r["stage1_ms"] + r["stage2_ms"] for r in records),
        },
    }

    for order_name, ordered in orderings.items():
        rows = []
        for size in cache_sizes:
            for policy in policies:
                print(f"  [{order_name}] {policy} @ {size} ...", flush=True)
                rows.append(
                    simulate_one(
                        ordered,
                        oracle_keys,
                        policy,
                        size,
                        threshold,
                        embedder_model,
                        embed_map,
                        hit_cost_ms,
                        random_seed,
                    )
                )
        dataset_payload[order_name] = rows

    return dataset_payload


def build_metadata(
    cache_sizes: list[int],
    policies: list[str],
    threshold: float,
    embedder_model: str,
    hit_cost_ms: int,
    shuffle_seed: int,
    random_seed: int,
    limit,
) -> dict:
    return {
        "cache_sizes": cache_sizes,
        "policies": policies,
        "shuffle_seed": shuffle_seed,
        "random_seed": random_seed,
        "similarity_threshold": threshold,
        "embedder_model": embedder_model,
        "hit_cost_ms": hit_cost_ms,
        "question_limit": limit,
        "access_patterns": ["sequential", "question_shuffled"],
        "shuffled_semantics": "Questions are replayed in a shuffled order; each question keeps its own measured stage-1/stage-2 cost.",
        # Which time bucket the cache shrinks, and which it leaves alone.
        # Mirrors cache_sim_summary.json so the same plotters read both.
        "stage_keys": {"cached": "planner", "uncached": "reasoner"},
        # Kept short: these become y-axis labels ("Planner Time (s)",
        # "Planner Speedup"), which clip at a 6.5in single-column width.
        "stage_labels": {"cached": "Planner", "uncached": "Reasoner"},
        "source": "simulate_rog_cache_timing.py",
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", action="append", default=[], metavar="[DATASET=]PATH",
                        help="run directory holding an uncached (`none`) policy with sidecars, "
                             "to source measured per-question timings from. Repeatable and "
                             "optionally dataset-qualified (RoG-webqsp=artifacts/...). A dataset "
                             "with no run still gets hit rates, with zeroed time columns.")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="use only the first N questions of each dataset")
    parser.add_argument("--policy-tag", default="none",
                        help="tag of the uncached run inside --run (default: none)")
    parser.add_argument("--cache-sizes", default=",".join(str(s) for s in DEFAULT_CACHE_SIZES))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("-t", "--threshold", type=float, default=0.90)
    parser.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--hit-cost-s", type=float, default=0.0,
                        help="modelled cost of serving a hit (embed + cosine scan). "
                             "0.0 reports the upper bound on savings; a real run measured ~0.10s")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0,
                        help="seed for the semantic_random eviction policy")
    parser.add_argument("--data-path", default="rmanluo")
    parser.add_argument("-d", "--datasets", default="RoG-webqsp",
                        help="comma-separated datasets; each becomes one entry under "
                             "'datasets' and therefore one comparison figure")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cache_sizes = _parse_int_list(args.cache_sizes)
    policies = _parse_str_list(args.policies)
    datasets = _parse_str_list(args.datasets)

    # --run PATH applies to the only dataset; --run DATASET=PATH is explicit.
    runs: dict = {}
    for entry in args.run:
        if "=" in entry:
            name, path = entry.split("=", 1)
            runs[name.strip()] = Path(path)
        elif len(datasets) == 1:
            runs[datasets[0]] = Path(entry)
        else:
            raise SystemExit(
                f"--run {entry!r} is ambiguous with {len(datasets)} datasets; "
                f"qualify it as DATASET=PATH"
            )

    payload = {
        "metadata": build_metadata(
            cache_sizes, policies, args.threshold, args.embedder_model,
            int(round(args.hit_cost_s * 1000)), args.shuffle_seed, args.random_seed,
            args.limit,
        ),
        "datasets": {},
    }

    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        run_dir = runs.get(dataset)
        if run_dir is not None:
            records, provenance = load_uncached_trace(run_dir, args.policy_tag)
            if args.limit is not None:
                records = records[: args.limit]
            provenance["timing_available"] = True
            print(f"loaded {len(records)} uncached questions from {run_dir}")
            print(f"  planner  {sum(r['stage1_ms'] for r in records) / 1000:8.1f}s (cacheable)")
            print(f"  reasoner {sum(r['stage2_ms'] for r in records) / 1000:8.1f}s (always paid)")
        else:
            records, provenance = load_questions_from_dataset(
                args.data_path, dataset, args.split, args.limit
            )
            print(f"loaded {len(records)} questions from {dataset}[{args.split}] "
                  f"(no uncached run: hit rates only, time columns zeroed)")
        print(f"  unique questions: {len(set(r['question'] for r in records))}")

        oracle_keys = {}
        if "semantic_oracle" in policies:
            oracle_keys = load_oracle_keys(records, args.data_path, dataset, args.split)

        payload["datasets"][dataset] = build_dataset_payload(
            records=records,
            provenance=provenance,
            oracle_keys=oracle_keys,
            cache_sizes=cache_sizes,
            policies=policies,
            threshold=args.threshold,
            embedder_model=args.embedder_model,
            hit_cost_ms=int(round(args.hit_cost_s * 1000)),
            shuffle_seed=args.shuffle_seed,
            random_seed=args.random_seed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)
        outfile.write("\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
