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

Speed:
  - Embeddings are precomputed in one batched pass over the dataset and
    reused across every (policy, capacity) run.
  - Cosine search is vectorized with numpy.
  - (policy, capacity) configs are run in parallel threads (numpy releases
    the GIL on matrix-vector products). Use --workers to control parallelism.

Usage:
    python simulate_cache.py [-d webqsp|lcquad|lcquad_test|...] [-n 500]
                             [-c 32,128,512,2048,inf]
                             [-p exact,semantic_lru,semantic_lfu,semantic_oracle]
                             [-t 0.90]
                             [--passes 1] [--workers 8]

Note: lcquad records have no populated `answer` field, so the
`semantic_oracle` policy degrades to all-miss on lcquad.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TOG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOG_DIR))

from question_cache import (  # noqa: E402
    PersistentQuestionCache,
    _USES_EMBEDDING,
    _normalize,
    _select_torch_device,
    extract_oracle_answer_key,
)

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
        os.chdir(TOG_DIR)
        from utils import prepare_dataset
        datas, qstr = prepare_dataset(dataset)
    finally:
        os.chdir(cwd)
    if limit is not None:
        datas = datas[: min(limit, len(datas))]
    return datas, qstr


class FastSimCache(PersistentQuestionCache):
    """Simulation-only subclass: precomputed embeddings + numpy-vectorized cosine.

    Behaviour matches PersistentQuestionCache; only performance differs:
      - `_embed` reads from a precomputed `embed_map` (no per-call SBERT call).
      - `_semantic_lookup` does a single numpy matmul over a stacked matrix
        of all currently-cached embeddings instead of a Python loop.
      - Per-hit prints are suppressed (would be 10k+ lines on big datasets).
    """

    def __init__(self, *args, embed_map=None, **kw):
        super().__init__(*args, **kw)
        self._embed_map = embed_map  # dict[normalized_question, np.ndarray(float32)]
        self._matrix_keys: list[str] = []
        self._matrix: "np.ndarray | None" = None
        self._matrix_dirty = True

    def _embed(self, q: str):
        if self._embed_map is not None:
            v = self._embed_map.get(_normalize(q))
            if v is not None:
                return v
        return super()._embed(q)

    def put(self, *args, **kw):
        super().put(*args, **kw)
        self._matrix_dirty = True

    def _evict_one(self):
        r = super()._evict_one()
        self._matrix_dirty = True
        return r

    def _rebuild_matrix(self):
        keys = list(self._embeddings.keys())
        if not keys:
            self._matrix = None
            self._matrix_keys = []
        else:
            self._matrix = np.stack(
                [np.asarray(self._embeddings[k], dtype=np.float32) for k in keys]
            )
            self._matrix_keys = keys
        self._matrix_dirty = False

    def _semantic_lookup(self, query_key, query_oracle_key=None, require_oracle=False):
        if not self._embeddings:
            return None
        if require_oracle:
            if not query_oracle_key:
                return None
            qset = {str(x) for x in query_oracle_key}
            if not qset:
                return None
        try:
            qv = self._embed(query_key)
        except Exception as e:
            print(f"[question_cache] embed failed, skipping semantic lookup: {e}")
            return None
        if self._matrix_dirty:
            self._rebuild_matrix()
        if self._matrix is None:
            return None
        qv = np.asarray(qv, dtype=np.float32)
        sims = self._matrix @ qv  # (N,) of cosines (L2-normalized embeddings)
        if require_oracle:
            allowed = np.fromiter(
                (
                    bool(self._oracle_keys.get(k))
                    and bool(qset.intersection(self._oracle_keys[k]))
                    for k in self._matrix_keys
                ),
                dtype=bool,
                count=len(self._matrix_keys),
            )
            sims = np.where(allowed, sims, -1.0)
        sims = np.where(sims >= self.similarity_threshold, sims, -1.0)
        idx = int(sims.argmax())
        if sims[idx] < 0:
            return None
        return self._matrix_keys[idx], float(sims[idx])

    # Suppress per-hit prints; too noisy at dataset scale.
    def get(self, question, oracle_key=None):
        key = _normalize(question)
        with self._lock:
            if key in self._store:
                self._touch(key)
                self.hits += 1
                self.exact_hits += 1
                return self._store[key]
            if self.policy in ("semantic_lru", "semantic_lfu"):
                sem = self._semantic_lookup(key)
                if sem is not None:
                    matched_key, _sim = sem
                    self._touch(matched_key)
                    self.hits += 1
                    if self.policy == "semantic_lfu":
                        self.semantic_lfu_hits += 1
                    else:
                        self.semantic_lru_hits += 1
                    return self._store[matched_key]
            elif self.policy == "semantic_oracle":
                sem = self._semantic_lookup(key, oracle_key, require_oracle=True)
                if sem is not None:
                    matched_key, _sim = sem
                    self._touch(matched_key)
                    self.hits += 1
                    self.semantic_oracle_hits += 1
                    return self._store[matched_key]
            self.misses += 1
            return None


def precompute_embeddings(datas, question_string, embedder_model, batch_size=128):
    """Encode every question once. Returns dict[normalized_question, np.float32 vec]."""
    questions = [_normalize(d[question_string]) for d in datas]
    unique = list(dict.fromkeys(questions))
    print(
        f"precomputing embeddings for {len(unique)} unique questions "
        f"({len(questions) - len(unique)} duplicates) ...",
        flush=True,
    )
    device = _select_torch_device()
    print(f"  embedder={embedder_model!r} device={device}", flush=True)
    t0 = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedder_model, device=device)
        embs = model.encode(
            unique,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embs = np.asarray(embs, dtype=np.float32)
    except Exception:
        # Fallback: HF transformers + mean-pool. Slower but no extra dep.
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = embedder_model if "/" in embedder_model else f"sentence-transformers/{embedder_model}"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()
        out_chunks = []
        with torch.no_grad():
            for i in range(0, len(unique), batch_size):
                batch = unique[i : i + batch_size]
                enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (out.last_hidden_state * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                v = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
                out_chunks.append(v.cpu().numpy().astype(np.float32))
        embs = np.concatenate(out_chunks, axis=0)
    elapsed = time.perf_counter() - t0
    print(f"  encoded {len(unique)} questions in {elapsed:.1f}s", flush=True)
    return {k: embs[i] for i, k in enumerate(unique)}


def simulate(datas, question_string, policy, capacity,
             similarity_threshold, embedder_model, passes,
             precomputed_oracle_keys=None, embed_map=None):
    cache = FastSimCache(
        path="",
        capacity=capacity,
        policy=policy,
        similarity_threshold=similarity_threshold,
        embedder_model=embedder_model,
        embed_map=embed_map,
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
    ap.add_argument("-t", "--similarity-threshold", type=float, default=0.9)
    ap.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--passes", type=int, default=1,
                    help="how many times to iterate the dataset (>=2 reveals exact-hit potential)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="parallel (policy, capacity) workers (threads)")
    ap.add_argument("--embed-batch-size", type=int, default=128)
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
        if n_with_keys == 0:
            print(f"  warning: dataset={args.dataset!r} has no extractable gold-answer keys; "
                  f"semantic_oracle will degrade to all-miss")

    embed_map = None
    if any(p in _USES_EMBEDDING for p in policies):
        embed_map = precompute_embeddings(
            datas, qstr, args.embedder_model, batch_size=args.embed_batch_size
        )

    tasks = [(p, c) for p in policies for c in capacities]
    results = {}

    def run_one(p, c):
        return (p, c), simulate(
            datas, qstr, p, c,
            args.similarity_threshold, args.embedder_model,
            args.passes, precomputed_oracle_keys, embed_map,
        )

    workers = max(1, min(args.workers, len(tasks)))
    print(f"running {len(tasks)} (policy, capacity) configs across {workers} thread(s) ...", flush=True)
    t_all = time.perf_counter()
    if workers == 1:
        for p, c in tasks:
            tag = f"policy={p:<16} capacity={('inf' if c >= 10**9 else c)}"
            print(f"  [start] {tag}", flush=True)
            (key, r) = run_one(p, c)
            results[key] = r
            print(f"  [done]  {tag}  hit_rate={fmt_pct(r['hit_rate'])}  wall={r['wall_s']}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_task = {ex.submit(run_one, p, c): (p, c) for (p, c) in tasks}
            for fut in as_completed(fut_to_task):
                (p, c) = fut_to_task[fut]
                tag = f"policy={p:<16} capacity={('inf' if c >= 10**9 else c)}"
                try:
                    key, r = fut.result()
                except Exception as e:
                    print(f"  [FAIL]  {tag}  {e!r}", flush=True)
                    raise
                results[key] = r
                print(f"  [done]  {tag}  hit_rate={fmt_pct(r['hit_rate'])}  wall={r['wall_s']}s", flush=True)
    print(f"all configs finished in {time.perf_counter() - t_all:.1f}s", flush=True)

    print_table(policies, capacities, results, len(datas), args.passes)


if __name__ == "__main__":
    main()
