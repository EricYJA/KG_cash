"""Hit-rate simulation for the RoG planner cache -- no LLM, no GPU, seconds not hours.

The RoG analogue of ToG's simulate_cache.py, and it reuses that file's FastSimCache
(precomputed embeddings + numpy-vectorized cosine) unchanged. It replays the
dataset's questions in order, doing get() then put() on a miss, and reports the
hit rate for each (policy, capacity).

This answers "how many planner beam searches would caching skip?" for free. It
says nothing about accuracy -- for that you need the real pipeline
(scripts/run_rog_cache_experiment.sh), which is why this exists: use it to pick a
threshold, then pay for the GPU run only at that threshold.

Usage (inside the rog-eval image, see run_rog_cache_sim.sh):
    python simulate_rog_cache.py -d RoG-webqsp --split test
    python simulate_rog_cache.py -t 0.80,0.85,0.90,0.95 -c 128,inf
"""

import argparse
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.environ.get("TOG_CACHE_DIR", "/togcache"))

from rog_question_cache import extract_oracle_answer_key  # noqa: E402
from simulate_cache import (  # noqa: E402  (ToG's simulator, reused as-is)
    FastSimCache,
    parse_capacity_list,
    precompute_embeddings,
)

POLICIES = ("exact", "semantic_lru", "semantic_lfu", "semantic_oracle")


def simulate(questions, oracle_keys, policy, capacity, threshold, embedder_model, embed_map):
    cache = FastSimCache(
        path="",
        capacity=capacity,
        policy=policy,
        similarity_threshold=threshold,
        embedder_model=embedder_model,
        embed_map=embed_map,
    )
    for i, question in enumerate(questions):
        ok = oracle_keys[i] if policy == "semantic_oracle" else None
        if cache.get(question, oracle_key=ok) is None:
            cache.put(question, ["DUMMY_RELATION_PATH"], oracle_key=ok)
    return cache.stats()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", default="rmanluo")
    ap.add_argument("-d", "--dataset", default="RoG-webqsp")
    ap.add_argument("--split", default="test")
    ap.add_argument("-c", "--capacities", default="128,512,inf")
    ap.add_argument("-p", "--policies", default="exact,semantic_lru,semantic_lfu,semantic_oracle")
    ap.add_argument("-t", "--thresholds", default="0.80,0.85,0.90,0.95")
    ap.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    dataset = load_dataset(os.path.join(args.data_path, args.dataset), split=args.split)
    questions = [d["question"] for d in dataset]
    oracle_keys = [extract_oracle_answer_key(d, args.dataset) for d in dataset]
    n_oracle = sum(1 for k in oracle_keys if k)
    print(f"loaded {len(questions)} questions from {args.dataset} [{args.split}]")
    print(f"  unique questions: {len(set(questions))}")
    print(f"  records with an extractable gold-answer key (for semantic_oracle): "
          f"{n_oracle}/{len(questions)}")

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in POLICIES:
            sys.exit(f"unknown policy {p!r} (choose from {POLICIES})")
    capacities = parse_capacity_list(args.capacities)
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    embed_map = precompute_embeddings(dataset, "question", args.embedder_model)

    cap_labels = [("inf" if c >= 10**9 else str(c)) for c in capacities]
    for threshold in thresholds:
        print()
        print(f"=== hit rate @ cosine threshold {threshold} "
              f"[N={len(questions)}, single pass, cold cache] ===")
        header = f"{'policy':<18}" + "".join(f"{c:>12}" for c in cap_labels)
        print(header)
        print("-" * len(header))
        for policy in policies:
            row = f"{policy:<18}"
            for capacity in capacities:
                stats = simulate(questions, oracle_keys, policy, capacity,
                                 threshold, args.embedder_model, embed_map)
                row += f"{100 * stats['hit_rate']:>11.1f}%"
            print(row)
        if "exact" in policies:
            print("(`exact` is threshold-independent: it is the repeated-question rate.)")


if __name__ == "__main__":
    main()
