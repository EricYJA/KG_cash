#!/usr/bin/env python3
"""Simulate the question cache across policies and capacities — no LLM, no SPARQL.

Iterates over the dataset's questions in their natural order. For each one,
it does a cache `get`; on miss, it does a `put` with a dummy chain (and the
gold-answer oracle_key when policy='semantic_oracle'). It then reports per-
(policy, capacity) hit / miss / hit_rate / breakdown.

This tells you, for each policy, how many ToG runs you would have skipped
on this dataset before paying any LLM/Virtuoso cost.

Policies:
  - exact            : key match only
  - semantic_lru     : exact + cosine >= threshold (LRU eviction)
  - semantic_lfu     : exact + cosine >= threshold (LFU eviction)
  - semantic_oracle  : exact + cosine >= threshold AND gold-answer overlap

Usage:
    python simulate_cache.py [-d webqsp] [-n 500]
                             [-c 32,128,512,2048,inf]
                             [-p exact,semantic_lru,semantic_lfu,semantic_oracle]
                             [-t 0.90]
                             [--passes 1]
"""

import argparse
import sys
import time
from pathlib import Path

TOG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOG_DIR))

from question_cache import PersistentQuestionCache, extract_oracle_answer_key  # noqa: E402

POLICIES = ("exact", "semantic_lru", "semantic_lfu", "semantic_oracle")


def parse_capacity_list(s: str):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("inf", "infinite", "unbounded"):
            out.append(10**9)
        else:
            out.append(int(tok))
    return out


def load_dataset(dataset: str, limit):
    cwd = Path.cwd()
    try:
        import os
        os.chdir(TOG_DIR)
        from utils import prepare_dataset
        datas, qstr = prepare_dataset(dataset)
    finally:
        os.chdir(cwd)
    if limit is not None:
        datas = datas[: min(limit, len(datas))]
    return datas, qstr


def simulate(datas, question_string, policy, capacity,
             similarity_threshold, embedder_model, passes,
             precomputed_oracle_keys=None):
    cache = PersistentQuestionCache(
        path="",  # in-memory only
        capacity=capacity,
        policy=policy,
        similarity_threshold=similarity_threshold,
        embedder_model=embedder_model,
    )
    total_lookups = 0
    t0 = time.perf_counter()
    for _ in range(passes):
        for i, data in enumerate(datas):
            question = data[question_string]
            ok = (precomputed_oracle_keys[i]
                  if (policy == "semantic_oracle" and precomputed_oracle_keys)
                  else None)
            chain = cache.get(question, oracle_key=ok)
            total_lookups += 1
            if chain is None:
                cache.put(question, ["DUMMY_CHAIN"], oracle_key=ok)
    elapsed = time.perf_counter() - t0
    s = cache.stats()
    s["lookups"] = total_lookups
    s["wall_s"] = round(elapsed, 2)
    return s


def fmt_pct(x):
    return f"{100*x:5.1f}%"


def print_table(policies, capacities, results, n_questions, passes):
    cap_labels = [("inf" if c >= 10**9 else str(c)) for c in capacities]
    print()
    print(f"=== Hit rate per (policy, capacity)  "
          f"[N={n_questions} × {passes} pass(es) = {n_questions*passes} lookups] ===")
    header = f"{'policy':<18}" + "".join(f"{c:>12}" for c in cap_labels)
    print(header)
    print("-" * len(header))
    for p in policies:
        row = f"{p:<18}"
        for c in capacities:
            r = results[(p, c)]
            row += f"{fmt_pct(r['hit_rate']):>12}"
        print(row)

    print()
    print("=== Hit breakdown per (policy, capacity) ===")
    for p in policies:
        print(f"\n[{p}]")
        print(f"  {'capacity':<10}{'hits':>8}{'exact':>8}{'sem_lru':>10}"
              f"{'sem_lfu':>10}{'sem_orac':>10}{'miss':>8}{'rate':>8}{'wall_s':>9}")
        for c, lbl in zip(capacities, cap_labels):
            r = results[(p, c)]
            print(f"  {lbl:<10}"
                  f"{r['hits']:>8}{r['exact_hits']:>8}"
                  f"{r['semantic_lru_hits']:>10}{r['semantic_lfu_hits']:>10}"
                  f"{r['semantic_oracle_hits']:>10}{r['misses']:>8}"
                  f"{fmt_pct(r['hit_rate']):>8}{r['wall_s']:>9}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--dataset", default="webqsp")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="number of questions to take from the dataset (default: all)")
    ap.add_argument("-c", "--capacities", default="32,128,512,2048,inf",
                    help="comma-separated capacities; 'inf' for unbounded")
    ap.add_argument("-p", "--policies", default="exact,semantic_lru,semantic_lfu,semantic_oracle",
                    help=f"comma-separated policies to test; choose from {POLICIES}")
    ap.add_argument("-t", "--similarity-threshold", type=float, default=0.85)
    ap.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--passes", type=int, default=1,
                    help="how many times to iterate the dataset (>=2 reveals exact-hit potential)")
    args = ap.parse_args()

    datas, qstr = load_dataset(args.dataset, args.limit)
    print(f"loaded {len(datas)} records from dataset={args.dataset!r}")

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    capacities = parse_capacity_list(args.capacities)
    for p in policies:
        if p not in POLICIES:
            sys.exit(f"unknown policy: {p!r} (choose from {POLICIES})")

    precomputed_oracle_keys = None
    if "semantic_oracle" in policies:
        precomputed_oracle_keys = [extract_oracle_answer_key(d, args.dataset) for d in datas]
        n_with_keys = sum(1 for k in precomputed_oracle_keys if k)
        print(f"semantic_oracle: {n_with_keys}/{len(datas)} records have an extractable gold-answer key")

    results = {}
    for p in policies:
        for c in capacities:
            tag = f"policy={p:<16} capacity={('inf' if c >= 10**9 else c)}"
            print(f"  running {tag} ...", flush=True)
            results[(p, c)] = simulate(
                datas, qstr, p, c,
                args.similarity_threshold, args.embedder_model,
                args.passes, precomputed_oracle_keys,
            )

    print_table(policies, capacities, results, len(datas), args.passes)


if __name__ == "__main__":
    main()
