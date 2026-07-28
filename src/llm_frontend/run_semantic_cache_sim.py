from __future__ import annotations

import argparse
import json
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cache_simulator import CacheSimResult, run_simulation


@dataclass
class SemanticCacheSimResult:
    policy: str
    cache_size: int
    threshold: float
    requests: int
    hits: int
    misses: int
    semantic_hits: int
    avg_entity_overlap: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "cache_size": self.cache_size,
            "threshold": self.threshold,
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "hit_rate": round(self.hit_rate, 4),
            "avg_entity_overlap": round(self.avg_entity_overlap, 4),
        }


def _compute_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def extract_question_sequence(
    traces: list[dict],
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[list[str], list[list[str]]]:
    if shuffle:
        rng = random.Random(seed)
        traces = list(traces)
        rng.shuffle(traces)
    questions: list[str] = []
    entities: list[list[str]] = []
    for trace in traces:
        q = trace.get("question", "").strip()
        if q:
            questions.append(q)
            entities.append(list(trace.get("llm_initial_frontier", [])))
    return questions, entities


def _simulate_semantic_oracle(
    questions: list[str],
    entities: list[list[str]],
    embeddings: dict[str, Any],
    cache_size: int,
    threshold: float,
) -> SemanticCacheSimResult:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    counts = Counter(questions)
    cached_qs = [q for q, _ in counts.most_common(cache_size)]
    exact_cache = set(cached_qs)
    cached_ents: dict[str, set[str]] = {q: set() for q in cached_qs}
    for q, ents in zip(questions, entities):
        if q in cached_ents:
            cached_ents[q].update(ents)

    if cached_qs:
        cached_vecs = np.array([embeddings[q] for q in cached_qs])
    else:
        cached_vecs = np.empty((0,))

    hits = 0
    semantic_hits = 0
    total_overlap = 0.0

    for q, ents in zip(questions, entities):
        if q in exact_cache:
            hits += 1
        elif len(cached_qs) > 0:
            q_vec = embeddings[q].reshape(1, -1)
            sims = cosine_similarity(q_vec, cached_vecs)[0]
            max_idx = int(np.argmax(sims))
            if sims[max_idx] >= threshold:
                hits += 1
                semantic_hits += 1
                total_overlap += _compute_jaccard(set(ents), cached_ents[cached_qs[max_idx]])

    avg_overlap = total_overlap / semantic_hits if semantic_hits > 0 else 0.0
    return SemanticCacheSimResult(
        policy="semantic_oracle",
        cache_size=cache_size,
        threshold=threshold,
        requests=len(questions),
        hits=hits,
        misses=len(questions) - hits,
        semantic_hits=semantic_hits,
        avg_entity_overlap=avg_overlap,
    )


def _simulate_semantic_lru(
    questions: list[str],
    entities: list[list[str]],
    embeddings: dict[str, Any],
    cache_size: int,
    threshold: float,
) -> SemanticCacheSimResult:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    cache: OrderedDict[str, list[str]] = OrderedDict()
    hits = 0
    semantic_hits = 0
    total_overlap = 0.0

    for q, ents in zip(questions, entities):
        if q in cache:
            cache.move_to_end(q)
            hits += 1
        else:
            matched = False
            if cache:
                cached_qs = list(cache.keys())
                cached_vecs = np.array([embeddings[cq] for cq in cached_qs])
                q_vec = embeddings[q].reshape(1, -1)
                sims = cosine_similarity(q_vec, cached_vecs)[0]
                max_idx = int(np.argmax(sims))
                if sims[max_idx] >= threshold:
                    matched_q = cached_qs[max_idx]
                    cache.move_to_end(matched_q)
                    hits += 1
                    semantic_hits += 1
                    total_overlap += _compute_jaccard(set(ents), set(cache[matched_q]))
                    matched = True
            if not matched:
                if len(cache) >= cache_size:
                    cache.popitem(last=False)
                cache[q] = list(ents)

    avg_overlap = total_overlap / semantic_hits if semantic_hits > 0 else 0.0
    return SemanticCacheSimResult(
        policy="semantic_lru",
        cache_size=cache_size,
        threshold=threshold,
        requests=len(questions),
        hits=hits,
        misses=len(questions) - hits,
        semantic_hits=semantic_hits,
        avg_entity_overlap=avg_overlap,
    )


def _simulate_semantic_lfu(
    questions: list[str],
    entities: list[list[str]],
    embeddings: dict[str, Any],
    cache_size: int,
    threshold: float,
) -> SemanticCacheSimResult:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    cache: dict[str, list[str]] = {}
    freq: Counter[str] = Counter()
    hits = 0
    semantic_hits = 0
    total_overlap = 0.0

    for q, ents in zip(questions, entities):
        if q in cache:
            freq[q] += 1
            hits += 1
        else:
            matched = False
            if cache:
                cached_qs = list(cache.keys())
                cached_vecs = np.array([embeddings[cq] for cq in cached_qs])
                q_vec = embeddings[q].reshape(1, -1)
                sims = cosine_similarity(q_vec, cached_vecs)[0]
                max_idx = int(np.argmax(sims))
                if sims[max_idx] >= threshold:
                    matched_q = cached_qs[max_idx]
                    freq[matched_q] += 1
                    hits += 1
                    semantic_hits += 1
                    total_overlap += _compute_jaccard(set(ents), set(cache[matched_q]))
                    matched = True
            if not matched:
                if len(cache) >= cache_size:
                    lfu_key = min(freq, key=lambda k: freq[k])
                    del cache[lfu_key]
                    del freq[lfu_key]
                cache[q] = list(ents)
                freq[q] = 1

    avg_overlap = total_overlap / semantic_hits if semantic_hits > 0 else 0.0
    return SemanticCacheSimResult(
        policy="semantic_lfu",
        cache_size=cache_size,
        threshold=threshold,
        requests=len(questions),
        hits=hits,
        misses=len(questions) - hits,
        semantic_hits=semantic_hits,
        avg_entity_overlap=avg_overlap,
    )


def _simulate_kg_enhanced_lru(
    questions: list[str],
    entities: list[list[str]],
    embeddings: dict[str, Any],
    cache_size: int,
    threshold: float,
) -> SemanticCacheSimResult:
    """KG-Enhanced Semantic Cache (Tran et al., IEEE 2024 doi 10780864).

    Defaults reflect the paper's reported setup:
      - embedder:  sentence-transformers/all-MiniLM-L6-v2
      - threshold: 0.8 cosine similarity (paper's empirical choice)
      - eviction:  LRU
      - extra gate: query and cached entry must share at least one
                   knowledge-graph entity. The paper builds the KG from
                   Wikipedia sentences associated with each prompt; here we
                   adapt it to KGQA by using the dataset's topic-entity set
                   (`llm_initial_frontier`) as the per-question KG signature.

    Hit rule:  exact-question OR
              (cosine >= threshold AND at least one shared KG entity).
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    cache: OrderedDict[str, list[str]] = OrderedDict()
    hits = 0
    semantic_hits = 0
    total_overlap = 0.0

    for q, ents in zip(questions, entities):
        if q in cache:
            cache.move_to_end(q)
            hits += 1
            continue
        matched = False
        if cache:
            cached_qs = list(cache.keys())
            cached_vecs = np.array([embeddings[cq] for cq in cached_qs])
            q_vec = embeddings[q].reshape(1, -1)
            sims = cosine_similarity(q_vec, cached_vecs)[0]
            qents = set(ents)
            best_idx = -1
            best_sim = -1.0
            for i, cq in enumerate(cached_qs):
                if sims[i] < threshold:
                    continue
                if not qents.intersection(cache[cq]):
                    continue
                if sims[i] > best_sim:
                    best_sim = float(sims[i])
                    best_idx = i
            if best_idx >= 0:
                matched_q = cached_qs[best_idx]
                cache.move_to_end(matched_q)
                hits += 1
                semantic_hits += 1
                total_overlap += _compute_jaccard(qents, set(cache[matched_q]))
                matched = True
        if not matched:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[q] = list(ents)

    avg_overlap = total_overlap / semantic_hits if semantic_hits > 0 else 0.0
    return SemanticCacheSimResult(
        policy="kg_enhanced_lru",
        cache_size=cache_size,
        threshold=threshold,
        requests=len(questions),
        hits=hits,
        misses=len(questions) - hits,
        semantic_hits=semantic_hits,
        avg_entity_overlap=avg_overlap,
    )


_SEMANTIC_POLICY_FNS = {
    "semantic_lru": _simulate_semantic_lru,
    "semantic_lfu": _simulate_semantic_lfu,
    "semantic_oracle": _simulate_semantic_oracle,
    "kg_enhanced_lru": _simulate_kg_enhanced_lru,
}


def _exact_oracle_hit_rate(questions: list[str], cache_size: int) -> float:
    counts = Counter(questions)
    cache = {q for q, _ in counts.most_common(cache_size)}
    hits = sum(1 for q in questions if q in cache)
    return hits / len(questions) if questions else 0.0


def _exact_lru_hit_rate(questions: list[str], cache_size: int) -> float:
    cache: OrderedDict[str, None] = OrderedDict()
    hits = 0
    for q in questions:
        if q in cache:
            cache.move_to_end(q)
            hits += 1
        else:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[q] = None
    return hits / len(questions) if questions else 0.0


def _exact_lfu_hit_rate(questions: list[str], cache_size: int) -> float:
    cache: dict[str, None] = {}
    freq: Counter[str] = Counter()
    hits = 0
    for q in questions:
        if q in cache:
            freq[q] += 1
            hits += 1
        else:
            if len(cache) >= cache_size:
                lfu_key = min(freq, key=lambda k: freq[k])
                del cache[lfu_key]
                del freq[lfu_key]
            cache[q] = None
            freq[q] = 1
    return hits / len(questions) if questions else 0.0


_EXACT_BASELINE_FNS = {
    "semantic_oracle": _exact_oracle_hit_rate,
    "semantic_lru": _exact_lru_hit_rate,
    "semantic_lfu": _exact_lfu_hit_rate,
    "kg_enhanced_lru": _exact_lru_hit_rate,
}


def compute_exact_baselines(
    questions: list[str],
    cache_sizes: list[int],
    policies: list[str],
) -> dict[tuple[str, int], float]:
    return {
        (policy, size): _EXACT_BASELINE_FNS[policy](questions, size)
        for policy in policies
        for size in cache_sizes
    }


def run_semantic_simulation(
    traces: list[dict],
    cache_sizes: list[int],
    policies: list[str],
    thresholds: list[float],
    shuffle: bool = False,
    seed: int | None = None,
) -> list[SemanticCacheSimResult]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic simulation requires sentence-transformers and scikit-learn.\n"
            "Install them with: pip install sentence-transformers scikit-learn"
        ) from exc

    questions, entities = extract_question_sequence(traces, shuffle=shuffle, seed=seed)
    unique_qs = list(set(questions))
    print(f"Computing embeddings for {len(unique_qs)} unique questions...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(unique_qs, show_progress_bar=True)
    embeddings: dict[str, Any] = dict(zip(unique_qs, vecs))

    results: list[SemanticCacheSimResult] = []
    for size in cache_sizes:
        for policy in policies:
            fn = _SEMANTIC_POLICY_FNS[policy]
            for threshold in thresholds:
                results.append(fn(questions, entities, embeddings, size, threshold))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate semantic cache policies over saved LLM trace JSONL."
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("results_iterative_webqsp_test_traces.jsonl"),
        help="JSONL file produced by run_webqsp_llm.py.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["lru", "lfu", "oracle"],
        default=["lru", "lfu", "oracle"],
        help="Entity-level cache policies.",
    )
    parser.add_argument(
        "--semantic-policies",
        nargs="+",
        choices=["semantic_lru", "semantic_lfu", "semantic_oracle", "kg_enhanced_lru"],
        default=["semantic_lru", "semantic_lfu", "semantic_oracle", "kg_enhanced_lru"],
        dest="semantic_policies",
        help="Question-level semantic cache policies.",
    )
    parser.add_argument(
        "--cache-sizes",
        nargs="+",
        type=int,
        default=[10, 50, 100, 500, 1000],
        dest="cache_sizes",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.99, 0.95, 0.90, 0.85, 0.80],
        help="Cosine similarity thresholds for semantic matching.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle traces before simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="semantic_cache_summary.json",
        help="Path to save simulation results as JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.traces)
    if not path.exists():
        raise SystemExit(f"Trace file not found: {path}")

    traces = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    print(f"Loaded {len(traces)} traces from {path}")

    entity_results: list[CacheSimResult] = run_simulation(
        traces, args.cache_sizes, args.policies, shuffle=args.shuffle, seed=args.seed
    )

    print(f"\n=== Entity-Level Cache Simulation ===")
    print(f"{'Policy':<8} {'Size':>6} {'Requests':>10} {'Hits':>8} {'Misses':>8} {'HitRate':>8}")
    print("-" * 54)
    for r in entity_results:
        print(
            f"{r.policy:<8} {r.cache_size:>6} {r.requests:>10} "
            f"{r.hits:>8} {r.misses:>8} {r.hit_rate:>8.2%}"
        )

    # Exact-match question baselines for gain comparison (same granularity as semantic sim).
    questions, _ = extract_question_sequence(traces, shuffle=args.shuffle, seed=args.seed)
    exact_baselines = compute_exact_baselines(questions, args.cache_sizes, args.semantic_policies)

    sem_results: list[SemanticCacheSimResult] = run_semantic_simulation(
        traces,
        args.cache_sizes,
        args.semantic_policies,
        args.thresholds,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print(f"\n=== Question-Level Semantic Cache Simulation ===")
    print(
        f"{'Policy':<16} {'Size':>6} {'Thresh':>7} {'Requests':>10} "
        f"{'Hits':>8} {'HitRate':>8} {'SemHits':>8} {'AvgOverlap':>11} {'Gain%':>7}"
    )
    print("-" * 93)
    sem_output_rows: list[dict] = []
    for r in sem_results:
        base_rate = exact_baselines.get((r.policy, r.cache_size))
        gain_pct = (r.hit_rate - base_rate) * 100 if base_rate is not None else float("nan")
        gain_str = f"{gain_pct:+.2f}%" if base_rate is not None else "  n/a"
        print(
            f"{r.policy:<16} {r.cache_size:>6} {r.threshold:>7.2f} {r.requests:>10} "
            f"{r.hits:>8} {r.hit_rate:>8.2%} {r.semantic_hits:>8} {r.avg_entity_overlap:>11.2%} {gain_str:>7}"
        )
        row = r.to_dict()
        row["exact_match_base_rate"] = round(base_rate, 4) if base_rate is not None else None
        row["gain_pct"] = round(gain_pct, 4) if base_rate is not None else None
        sem_output_rows.append(row)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "entity_level": [r.to_dict() for r in entity_results],
            "question_level_semantic": sem_output_rows,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
