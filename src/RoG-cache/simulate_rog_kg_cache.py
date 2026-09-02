#!/usr/bin/env python3
"""KG-request cache simulation for RoG -- the direct analogue of ToG's cache_simulator.py.

This is about caching *one-hop neighbourhood lookups*, not questions. It is the
RoG counterpart of cache_sim_webqsp_comparison.pdf, and has nothing to do with
the semantic question cache in simulate_rog_cache_timing.py.

Where the requests come from. RoG answers a question by taking the relation
paths its planner predicted and walking them over the KG with bfs_with_rule
(ref_KG_projects/RoG/src/utils/graph_utils.py:16). The inner loop of that walk is

    for neighbor in graph.neighbors(current_node):

which is exactly one one-hop neighbourhood lookup, keyed by the entity being
expanded -- the same cacheable unit as ToG's relation_lookup_head/tail. This
script replays that walk with the planner's real predicted paths and records
every expansion, producing a request stream the existing cache simulator can
replay under LRU/LFU/FIFO/Random/Belady/Oracle.

The one framing choice worth being explicit about. As shipped, RoG is handed a
per-question subgraph inside each HuggingFace example, so `neighbors(X)` is
scoped to that example and there is by construction nothing to share between
questions. That is an artifact of how the benchmark packages its data, not of
the algorithm: the entities are Freebase MIDs and the one-hop neighbourhood of a
MID is a global fact. This simulation therefore keys requests by entity alone,
i.e. it measures the cache RoG *would* have if it retrieved from a shared KG the
way ToG does. That is the only framing under which the two systems' figures are
comparable, and it is the question the figure is meant to answer.

What it does not model: latency. RoG's lookups are in-memory networkx calls with
no measured duration, so the time columns are zero and only the hit-rate figure
is meaningful. ToG's durations came from real SPARQL round trips.

Usage:
    python simulate_rog_kg_cache.py --predictions <none-run predictions_3_False.jsonl>
    python simulate_rog_kg_cache.py --limit 400 --cache-sizes 10,50,100,500,1000
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import datasets_compat  # noqa: F401  (read caches written by a newer datasets)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ToG-cache" / "ToG"))

from cache_simulator import (  # noqa: E402
    DEFAULT_RANDOM_REPEATS,
    KGRequest,
    SUPPORTED_POLICIES,
    run_simulation_from_requests,
)

DEFAULT_POLICIES = ("lru", "lfu", "oracle")
DEFAULT_CACHE_SIZES = (10, 50, 100, 500, 1000)
# Guards against bfs_with_rule's missing visited-set: it enqueues paths, not
# nodes, so a dense subgraph with a 2-hop rule can expand combinatorially.
DEFAULT_MAX_EXPANSIONS = 20000


def build_adjacency(triples) -> tuple[dict, dict]:
    """Mirror RoG's build_graph: an undirected nx.Graph, one relation per node pair.

    nx.Graph collapses parallel edges, so a later triple between the same two
    entities overwrites the earlier one's relation. Replicated here so the walk
    sees exactly the graph RoG's walk sees.
    """
    adjacency: dict = defaultdict(set)
    edge_relation: dict = {}
    for triple in triples:
        head, relation, tail = triple
        relation = relation.strip()
        edge_relation[(head, tail)] = relation
        edge_relation[(tail, head)] = relation
        adjacency[head].add(tail)
        adjacency[tail].add(head)
    return adjacency, edge_relation


def replay_bfs(adjacency, edge_relation, start_node, target_rule, expansions, max_expansions):
    """RoG's bfs_with_rule, recording each neighbourhood expansion it performs.

    Returns the number of expansions; `expansions` collects the expanded entity
    ids in issue order. Path bookkeeping is reduced to a depth counter because
    only the sequence of lookups matters here, not the paths returned.
    """
    queue = deque([(start_node, 0)])
    count = 0
    while queue:
        current_node, depth = queue.popleft()
        if depth >= len(target_rule):
            continue
        # The one-hop neighbourhood lookup. Recorded even when the node is absent
        # from this example's subgraph: against a real KG that is still a request,
        # it just comes back empty.
        expansions.append(current_node)
        count += 1
        if count >= max_expansions:
            return count, True
        wanted = target_rule[depth]
        for neighbor in adjacency.get(current_node, ()):
            if edge_relation.get((current_node, neighbor)) != wanted:
                continue
            queue.append((neighbor, depth + 1))
    return count, False


def _key(entity_id: str) -> str:
    """Same shape as cache_simulator._kg_key, so both summaries key alike."""
    return json.dumps(
        {"operation": "neighbor_expand", "input": {"entity_id": entity_id}},
        sort_keys=True,
        separators=(",", ":"),
    )


def build_request_blocks(dataset, predictions: dict, max_expansions: int):
    """One block of KG requests per question, in the order RoG would issue them."""
    blocks: list[list[KGRequest]] = []
    stats = {"questions": 0, "no_prediction": 0, "truncated": 0, "empty": 0}
    for row in dataset:
        rules = predictions.get(row["id"])
        if not rules:
            stats["no_prediction"] += 1
            continue
        stats["questions"] += 1
        adjacency, edge_relation = build_adjacency(row["graph"])
        expansions: list[str] = []
        truncated = False
        for rule in rules:
            if not rule:
                continue
            for start in row["q_entity"]:
                _, hit_cap = replay_bfs(
                    adjacency, edge_relation, start, rule, expansions, max_expansions
                )
                truncated = truncated or hit_cap
        if truncated:
            stats["truncated"] += 1
        if not expansions:
            stats["empty"] += 1
            continue
        blocks.append([KGRequest(key=_key(e), duration_ms=0, operation="neighbor_expand")
                       for e in expansions])
    return blocks, stats


def flatten(blocks) -> list[KGRequest]:
    return [request for block in blocks for request in block]


def load_predictions(path: Path) -> dict:
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            predictions[record["id"]] = record.get("prediction") or []
    return predictions


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", action="append", default=[], metavar="[DATASET=]PATH",
                    required=True,
                    help="stage-1 predictions_*.jsonl from an uncached run (supplies the "
                         "relation paths RoG actually walked). Repeatable and optionally "
                         "dataset-qualified (RoG-cwq=artifacts/...) when simulating several.")
    ap.add_argument("--data-path", default="rmanluo")
    ap.add_argument("-d", "--datasets", default="RoG-webqsp",
                    help="comma-separated datasets; each becomes one entry under 'datasets' "
                         "and therefore one comparison figure")
    ap.add_argument("--split", default="test")
    ap.add_argument("-n", "--limit", type=int, default=None)
    ap.add_argument("--cache-sizes", default=",".join(str(s) for s in DEFAULT_CACHE_SIZES))
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS)
    ap.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cache_sizes = _parse_int_list(args.cache_sizes)
    policies = _parse_str_list(args.policies)
    datasets = _parse_str_list(args.datasets)
    bad = [p for p in policies if p not in SUPPORTED_POLICIES]
    if bad:
        raise SystemExit(f"unsupported policies {bad}; choose from {list(SUPPORTED_POLICIES)}")

    # --predictions PATH applies to the only dataset; DATASET=PATH is explicit.
    prediction_paths: dict = {}
    for entry in args.predictions:
        if "=" in entry:
            name, path = entry.split("=", 1)
            prediction_paths[name.strip()] = Path(path)
        elif len(datasets) == 1:
            prediction_paths[datasets[0]] = Path(entry)
        else:
            raise SystemExit(
                f"--predictions {entry!r} is ambiguous with {len(datasets)} datasets; "
                f"qualify it as DATASET=PATH"
            )
    missing = [d for d in datasets if d not in prediction_paths]
    if missing:
        raise SystemExit(f"no --predictions given for {missing}")

    from datasets import load_dataset

    payload_datasets: dict = {}
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        data = load_dataset(os.path.join(args.data_path, dataset), split=args.split)
        if args.limit is not None:
            data = data.select(range(min(args.limit, len(data))))

        predictions = load_predictions(prediction_paths[dataset])
        print(f"loaded {len(predictions)} predicted path sets from {prediction_paths[dataset]}")

        blocks, stats = build_request_blocks(data, predictions, args.max_expansions)
        requests = flatten(blocks)
        unique = len({r.key for r in requests})
        print(f"{stats['questions']} questions replayed ({stats['no_prediction']} without a "
              f"prediction, {stats['empty']} with no expansion)")
        if stats["truncated"]:
            print(f"[warn] {stats['truncated']} questions hit the {args.max_expansions} "
                  f"expansion cap; their tails are missing")
        if not requests:
            raise SystemExit(f"no KG requests produced for {dataset}")
        print(f"{len(requests)} one-hop lookups, {unique} unique entities "
              f"-> reuse ceiling {100 * (1 - unique / len(requests)):.2f}%")

        shuffled_blocks = copy.copy(blocks)
        random.Random(args.shuffle_seed).shuffle(shuffled_blocks)
        breakdown = {"kg": 0, "llm": 0, "other": 0, "total": 0}

        dataset_payload = {
            "predictions_path": str(prediction_paths[dataset]),
            "request_count": len(requests),
            "unique_entities": unique,
            "questions": stats["questions"],
            "time_breakdown_ms": breakdown,
            "timing_available": False,
        }
        for name, stream in {"sequential": requests,
                             "shuffled": flatten(shuffled_blocks)}.items():
            dataset_payload[name] = [
                result.to_dict()
                for result in run_simulation_from_requests(
                    requests=stream, breakdown=breakdown, cache_sizes=cache_sizes,
                    policies=policies, random_seed=args.random_seed,
                    random_repeats=args.random_repeats,
                )
            ]
        payload_datasets[dataset] = dataset_payload

    payload = {
        "metadata": {
            "cache_sizes": cache_sizes,
            "policies": policies,
            "shuffle_seed": args.shuffle_seed,
            "random_seed": args.random_seed,
            "random_repeats": args.random_repeats,
            "question_limit": args.limit,
            "cache_unit": "one-hop neighbourhood lookup, keyed by entity id",
            "access_patterns": ["sequential", "request_block_shuffled"],
            "shuffled_semantics": "Question order is shuffled; the KG access order inside a question is preserved.",
            "stage_keys": {"cached": "kg", "uncached": "llm"},
            "stage_labels": {"cached": "KG", "uncached": "LLM"},
            "timing_note": "RoG's lookups are in-memory networkx calls with no measured "
                           "duration; time columns are zero and only hit rate is meaningful.",
            "source": "simulate_rog_kg_cache.py",
        },
        "datasets": payload_datasets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
