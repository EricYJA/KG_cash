from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass


@dataclass
class CacheSimResult:
    policy: str
    cache_size: int
    requests: int
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "cache_size": self.cache_size,
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


def _simulate_lru(access_sequence: list[str], cache_size: int) -> tuple[int, int]:
    cache: OrderedDict[str, None] = OrderedDict()
    hits = 0
    for entity_id in access_sequence:
        if entity_id in cache:
            cache.move_to_end(entity_id)
            hits += 1
        else:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[entity_id] = None
    return hits, len(access_sequence) - hits


def _simulate_lfu(access_sequence: list[str], cache_size: int) -> tuple[int, int]:
    cache: dict[str, None] = {}
    freq: Counter[str] = Counter()
    hits = 0
    for entity_id in access_sequence:
        if entity_id in cache:
            freq[entity_id] += 1
            hits += 1
        else:
            if len(cache) >= cache_size:
                lfu_key = min(freq, key=lambda k: freq[k])
                del cache[lfu_key]
                del freq[lfu_key]
            cache[entity_id] = None
            freq[entity_id] = 1
    return hits, len(access_sequence) - hits


def _simulate_oracle(access_sequence: list[str], cache_size: int) -> tuple[int, int]:
    freq = Counter(access_sequence)
    preloaded = {eid for eid, _ in freq.most_common(cache_size)}
    hits = sum(1 for eid in access_sequence if eid in preloaded)
    return hits, len(access_sequence) - hits


def _run_policy(
    policy: str,
    access_sequence: list[str],
    cache_size: int,
) -> CacheSimResult:
    if policy == "lru":
        hits, misses = _simulate_lru(access_sequence, cache_size)
    elif policy == "lfu":
        hits, misses = _simulate_lfu(access_sequence, cache_size)
    elif policy == "oracle":
        hits, misses = _simulate_oracle(access_sequence, cache_size)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return CacheSimResult(
        policy=policy,
        cache_size=cache_size,
        requests=len(access_sequence),
        hits=hits,
        misses=misses,
    )


def extract_access_sequence(traces: list[dict]) -> list[str]:
    sequence: list[str] = []
    for trace in traces:
        for step in trace.get("llm_kg_queries", []):
            sequence.extend(step.get("input_frontier", []))
    return sequence


def run_simulation(
    traces: list[dict],
    cache_sizes: list[int],
    policies: list[str],
) -> list[CacheSimResult]:
    sequence = extract_access_sequence(traces)
    results: list[CacheSimResult] = []
    for size in cache_sizes:
        for policy in policies:
            results.append(_run_policy(policy, sequence, size))
    return results
