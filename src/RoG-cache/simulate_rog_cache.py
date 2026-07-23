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


def _parse_limit(value):
    """None/'all'/'' -> no limit; otherwise a positive int. Mirrors --limit all."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("", "all", "none"):
        return None
    try:
        limit = int(text)
    except ValueError:
        raise SystemExit(f"--limit: expected a positive integer or 'all', got {value!r}")
    if limit <= 0:
        raise SystemExit(f"--limit: expected a positive integer or 'all', got {value!r}")
    return limit


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
    ap.add_argument("-n", "--limit", default=None,
                    help="use only the first N questions, or 'all' (default: all)")
    ap.add_argument("--out", default=None,
                    help="also write the result table to this file (still printed to stdout)")
    args = ap.parse_args()

    limit = _parse_limit(args.limit)

    dataset = load_dataset(os.path.join(args.data_path, args.dataset), split=args.split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
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

    lines: list[str] = []

    def emit(line=""):
        print(line)
        lines.append(line)

    cap_labels = [("inf" if c >= 10**9 else str(c)) for c in capacities]
    for threshold in thresholds:
        emit()
        emit(f"=== hit rate @ cosine threshold {threshold} "
             f"[N={len(questions)}, single pass, cold cache] ===")
        header = f"{'policy':<18}" + "".join(f"{c:>12}" for c in cap_labels)
        emit(header)
        emit("-" * len(header))
        for policy in policies:
            row = f"{policy:<18}"
            for capacity in capacities:
                stats = simulate(questions, oracle_keys, policy, capacity,
                                 threshold, args.embedder_model, embed_map)
                row += f"{100 * stats['hit_rate']:>11.1f}%"
            emit(row)
        if "exact" in policies:
            emit("(`exact` is threshold-independent: it is the repeated-question rate.)")

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n>>> table written to {args.out}")


if __name__ == "__main__":
    main()
