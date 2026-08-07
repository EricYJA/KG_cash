from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path


KG_EVENT_TYPE = "KG"
LLM_EVENT_TYPE = "LLM"
OTHER_EVENT_TYPE = "OTHER"
DEFAULT_RANDOM_REPEATS = 5
DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[1] / "output" / "traces"
DEFAULT_TRACE_FILES = {
    "WebQSP": DEFAULT_TRACE_DIR / "tog_trace_webqsp.json",
    "CWQ": DEFAULT_TRACE_DIR / "tog_trace_cwq.json",
}
DEFAULT_COMBINED_OUTPUT = Path(__file__).resolve().parents[1] / "output" / "cache_sim_summary.json"


@dataclass(frozen=True)
class KGRequest:
    key: str
    duration_ms: int
    operation: str


@dataclass
class CacheSimResult:
    policy: str
    cache_size: int
    requests: int
    hits: int
    misses: int
    kg_base_ms: int
    kg_simulated_ms: int
    llm_ms: int
    other_ms: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests > 0 else 0.0

    @property
    def total_base_ms(self) -> int:
        return self.kg_base_ms + self.llm_ms + self.other_ms

    @property
    def total_simulated_ms(self) -> int:
        return self.kg_simulated_ms + self.llm_ms + self.other_ms

    @property
    def saved_ms(self) -> int:
        return self.total_base_ms - self.total_simulated_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "cache_size": self.cache_size,
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "time_breakdown_ms": {
                "kg_base": self.kg_base_ms,
                "kg_simulated": self.kg_simulated_ms,
                "llm": self.llm_ms,
                "other": self.other_ms,
                "total_base": self.total_base_ms,
                "total_simulated": self.total_simulated_ms,
                "saved": self.saved_ms,
            },
        }


def load_traces(path: str | Path) -> list[dict]:
    trace_path = Path(path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    suffix = trace_path.suffix.lower()
    with trace_path.open("r", encoding="utf-8") as infile:
        if suffix == ".jsonl":
            return [json.loads(line) for line in infile if line.strip()]
        if suffix == ".json":
            payload = json.load(infile)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {trace_path}")
            return payload
    raise ValueError(f"Unsupported trace format for {trace_path}; expected .json or .jsonl")


def _event_duration_ms(event: dict) -> int:
    return int(event.get("duration_ms", 0) or 0)


def _kg_key(event: dict) -> str:
    return json.dumps(
        {
            "operation": event["operation"],
            "input": event.get("input", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _ordered_events(traces: list[dict]) -> list[dict]:
    ordered: list[dict] = []
    for trace in traces:
        ordered.extend(trace.get("events", []))
    return ordered


def extract_kg_requests(traces: list[dict]) -> list[KGRequest]:
    requests: list[KGRequest] = []
    for event in _ordered_events(traces):
        if event.get("type") != KG_EVENT_TYPE:
            continue
        requests.append(
            KGRequest(
                key=_kg_key(event),
                duration_ms=_event_duration_ms(event),
                operation=str(event.get("operation", "")),
            )
        )
    return requests


def extract_kg_request_blocks(traces: list[dict]) -> list[list[KGRequest]]:
    blocks: list[list[KGRequest]] = []
    for trace in traces:
        block: list[KGRequest] = []
        for event in trace.get("events", []):
            if event.get("type") != KG_EVENT_TYPE:
                continue
            block.append(
                KGRequest(
                    key=_kg_key(event),
                    duration_ms=_event_duration_ms(event),
                    operation=str(event.get("operation", "")),
                )
            )
        if block:
            blocks.append(block)
    return blocks


def flatten_request_blocks(blocks: list[list[KGRequest]]) -> list[KGRequest]:
    return [request for block in blocks for request in block]


def extract_time_breakdown(traces: list[dict]) -> dict[str, int]:
    kg_ms = 0
    llm_ms = 0
    other_ms = 0
    for event in _ordered_events(traces):
        duration_ms = _event_duration_ms(event)
        event_type = event.get("type")
        if event_type == KG_EVENT_TYPE:
            kg_ms += duration_ms
        elif event_type == LLM_EVENT_TYPE:
            llm_ms += duration_ms
        else:
            other_ms += duration_ms
    return {
        "kg": kg_ms,
        "llm": llm_ms,
        "other": other_ms,
        "total": kg_ms + llm_ms + other_ms,
    }


# Each simulator makes a single pass over the request stream and returns
# (hits, misses, kg_simulated_ms) together, so hit accounting and the served-time
# accounting can never drift apart. `kg_simulated_ms` is the summed duration of
# the requests that miss; hits are assumed to be served for free.


def _no_cache(requests: list[KGRequest]) -> tuple[int, int, int]:
    return 0, len(requests), sum(request.duration_ms for request in requests)


def _simulate_recency(
    requests: list[KGRequest], cache_size: int, *, touch_on_hit: bool
) -> tuple[int, int, int]:
    """LRU when `touch_on_hit`, FIFO otherwise -- the two differ only in that."""
    cache: OrderedDict[str, None] = OrderedDict()
    hits = 0
    kg_simulated_ms = 0
    for request in requests:
        if request.key in cache:
            hits += 1
            if touch_on_hit:
                cache.move_to_end(request.key)
            continue
        kg_simulated_ms += request.duration_ms
        if len(cache) >= cache_size:
            cache.popitem(last=False)
        cache[request.key] = None
    return hits, len(requests) - hits, kg_simulated_ms


def _simulate_lfu(requests: list[KGRequest], cache_size: int) -> tuple[int, int, int]:
    cache: dict[str, None] = {}
    freq: Counter[str] = Counter()
    hits = 0
    kg_simulated_ms = 0
    for request in requests:
        if request.key in cache:
            freq[request.key] += 1
            hits += 1
            continue
        kg_simulated_ms += request.duration_ms
        if len(cache) >= cache_size:
            lfu_key = min(freq, key=lambda key: freq[key])
            del cache[lfu_key]
            del freq[lfu_key]
        cache[request.key] = None
        freq[request.key] = 1
    return hits, len(requests) - hits, kg_simulated_ms


def _simulate_random_once(
    requests: list[KGRequest], cache_size: int, seed: int
) -> tuple[int, int, int]:
    rng = random.Random(seed)
    cache: dict[str, None] = {}
    # Parallel key list so eviction is an O(1) swap-with-last instead of an O(n)
    # reservoir walk over the dict.
    keys: list[str] = []
    hits = 0
    kg_simulated_ms = 0
    for request in requests:
        if request.key in cache:
            hits += 1
            continue
        kg_simulated_ms += request.duration_ms
        if len(cache) >= cache_size:
            victim_index = rng.randrange(len(keys))
            victim = keys[victim_index]
            keys[victim_index] = keys[-1]
            keys.pop()
            del cache[victim]
        cache[request.key] = None
        keys.append(request.key)
    return hits, len(requests) - hits, kg_simulated_ms


def _simulate_random(
    requests: list[KGRequest], cache_size: int, seed: int, repeats: int
) -> tuple[int, int, int]:
    """Random replacement over `repeats` seeds, reporting the *representative*
    run -- the one whose hit count lands closest to the mean.

    Averaging the three numbers independently would let them contradict each
    other (a mean miss count that does not match the mean served time, once
    both are rounded), which then shows up as a hit rate and a `saved` figure
    that disagree in the plots. Picking a single real run keeps the record
    internally consistent while still not letting one unlucky draw decide it.
    """
    runs = max(repeats, 1)
    trials = [_simulate_random_once(requests, cache_size, seed + offset) for offset in range(runs)]
    mean_hits = sum(trial[0] for trial in trials) / runs
    return min(trials, key=lambda trial: (abs(trial[0] - mean_hits), trial[0]))


def _next_use_table(requests: list[KGRequest]) -> list[int]:
    """next_use[i] = index of the next request for the same key, else len(requests)."""
    horizon = len(requests)
    next_use = [horizon] * horizon
    last_seen: dict[str, int] = {}
    for index in range(horizon - 1, -1, -1):
        key = requests[index].key
        next_use[index] = last_seen.get(key, horizon)
        last_seen[key] = index
    return next_use


def _simulate_belady(requests: list[KGRequest], cache_size: int) -> tuple[int, int, int]:
    """Belady's MIN: evict whatever is reused farthest in the future.

    This is the true offline optimum and therefore the honest upper bound on
    what any online policy could achieve. It is *not* the same thing as the
    `oracle` policy below, which only preloads the globally hottest keys.
    """
    horizon = len(requests)
    next_use = _next_use_table(requests)
    cached: dict[str, int] = {}  # key -> index of its next use
    hits = 0
    kg_simulated_ms = 0
    for index, request in enumerate(requests):
        if request.key in cached:
            hits += 1
            cached[request.key] = next_use[index]
            continue
        kg_simulated_ms += request.duration_ms
        if next_use[index] >= horizon:
            continue  # never referenced again: admitting it can only displace something useful
        if len(cached) >= cache_size:
            victim = max(cached, key=lambda key: cached[key])
            if cached[victim] <= next_use[index]:
                continue  # everything resident is reused sooner; decline admission
            del cached[victim]
        cached[request.key] = next_use[index]
    return hits, len(requests) - hits, kg_simulated_ms


def _simulate_oracle(requests: list[KGRequest], cache_size: int) -> tuple[int, int, int]:
    """Static preload of the globally most-frequent keys; never evicts."""
    freq = Counter(request.key for request in requests)
    preloaded = {key for key, _ in freq.most_common(cache_size)}
    hits = 0
    kg_simulated_ms = 0
    for request in requests:
        if request.key in preloaded:
            hits += 1
        else:
            kg_simulated_ms += request.duration_ms
    return hits, len(requests) - hits, kg_simulated_ms


# policy name -> (requests, cache_size, random_seed, random_repeats) -> (hits, misses, ms)
POLICY_SIMULATORS = {
    "lru":    lambda reqs, size, seed, reps: _simulate_recency(reqs, size, touch_on_hit=True),
    "fifo":   lambda reqs, size, seed, reps: _simulate_recency(reqs, size, touch_on_hit=False),
    "lfu":    lambda reqs, size, seed, reps: _simulate_lfu(reqs, size),
    "random": lambda reqs, size, seed, reps: _simulate_random(reqs, size, seed, reps),
    "belady": lambda reqs, size, seed, reps: _simulate_belady(reqs, size),
    "oracle": lambda reqs, size, seed, reps: _simulate_oracle(reqs, size),
}

SUPPORTED_POLICIES = tuple(POLICY_SIMULATORS)


def _simulate_kg_time(
    requests: list[KGRequest],
    policy: str,
    cache_size: int,
    random_seed: int = 0,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
) -> tuple[int, int, int]:
    try:
        simulator = POLICY_SIMULATORS[policy]
    except KeyError:
        raise ValueError(f"Unknown policy: {policy}") from None
    if cache_size <= 0:
        return _no_cache(requests)
    return simulator(requests, cache_size, random_seed, random_repeats)


def run_simulation(
    traces: list[dict],
    cache_sizes: list[int],
    policies: list[str],
    random_seed: int = 0,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
) -> list[CacheSimResult]:
    invalid_policies = [policy for policy in policies if policy not in SUPPORTED_POLICIES]
    if invalid_policies:
        raise ValueError(f"Unsupported policies: {', '.join(invalid_policies)}")

    breakdown = extract_time_breakdown(traces)
    requests = extract_kg_requests(traces)
    return run_simulation_from_requests(
        requests=requests,
        breakdown=breakdown,
        cache_sizes=cache_sizes,
        policies=policies,
        random_seed=random_seed,
        random_repeats=random_repeats,
    )


def run_simulation_from_requests(
    requests: list[KGRequest],
    breakdown: dict[str, int],
    cache_sizes: list[int],
    policies: list[str],
    random_seed: int = 0,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
) -> list[CacheSimResult]:
    invalid_policies = [policy for policy in policies if policy not in SUPPORTED_POLICIES]
    if invalid_policies:
        raise ValueError(f"Unsupported policies: {', '.join(invalid_policies)}")

    results: list[CacheSimResult] = []
    for size in cache_sizes:
        for policy in policies:
            hits, misses, kg_simulated_ms = _simulate_kg_time(
                requests, policy, size, random_seed, random_repeats
            )
            results.append(
                CacheSimResult(
                    policy=policy,
                    cache_size=size,
                    requests=len(requests),
                    hits=hits,
                    misses=misses,
                    kg_base_ms=breakdown["kg"],
                    kg_simulated_ms=kg_simulated_ms,
                    llm_ms=breakdown["llm"],
                    other_ms=breakdown["other"],
                )
            )
    return results


def build_combined_summary(
    trace_files: dict[str, Path],
    cache_sizes: list[int],
    policies: list[str],
    shuffle_seed: int,
    random_seed: int = 0,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "metadata": {
            "cache_sizes": cache_sizes,
            "policies": policies,
            "shuffle_seed": shuffle_seed,
            "random_seed": random_seed,
            "random_repeats": random_repeats,
            "access_patterns": ["sequential", "request_block_shuffled"],
            "shuffled_semantics": "Question/request traces are shuffled; internal KG access order is preserved.",
            # Which time bucket the cache shrinks, and which it leaves alone.
            # Consumed by plot_cache_time_breakdown.py so the same plotter works
            # on the RoG summary, where the cached stage is the planner LLM call.
            "stage_keys": {"cached": "kg", "uncached": "llm"},
            "stage_labels": {"cached": "KG", "uncached": "LLM"},
        },
        "datasets": {},
    }

    datasets = payload["datasets"]
    assert isinstance(datasets, dict)

    for dataset, trace_path in trace_files.items():
        traces = load_traces(trace_path)
        breakdown = extract_time_breakdown(traces)
        request_blocks = extract_kg_request_blocks(traces)
        sequential_requests = flatten_request_blocks(request_blocks)
        shuffled_blocks = copy.copy(request_blocks)
        rng = random.Random(shuffle_seed)
        rng.shuffle(shuffled_blocks)
        shuffled_requests = flatten_request_blocks(shuffled_blocks)

        datasets[dataset] = {
            "trace_path": str(trace_path),
            "request_count": len(sequential_requests),
            "time_breakdown_ms": breakdown,
            "sequential": [
                result.to_dict()
                for result in run_simulation_from_requests(
                    requests=sequential_requests,
                    breakdown=breakdown,
                    cache_sizes=cache_sizes,
                    policies=policies,
                    random_seed=random_seed,
                    random_repeats=random_repeats,
                )
            ],
            "shuffled": [
                result.to_dict()
                for result in run_simulation_from_requests(
                    requests=shuffled_requests,
                    breakdown=breakdown,
                    cache_sizes=cache_sizes,
                    policies=policies,
                    random_seed=random_seed,
                    random_repeats=random_repeats,
                )
            ],
        }

    return payload


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline cache simulation on ToG traces.")
    parser.add_argument("trace_path", nargs="?", help="path to trace .jsonl or pretty .json file")
    parser.add_argument("--cache-sizes", default="0,1,2,4,8,16,32", help="comma-separated cache sizes")
    parser.add_argument(
        "--policies",
        default="lru,lfu,fifo,random,belady,oracle",
        help=f"comma-separated policies from: {', '.join(SUPPORTED_POLICIES)}",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty print the simulation results as JSON")
    parser.add_argument("--output", type=Path, help="write simulation JSON to this path")
    parser.add_argument("--combined", action="store_true", help="simulate WebQSP and CWQ into one JSON file")
    parser.add_argument("--webqsp-trace", type=Path, default=DEFAULT_TRACE_FILES["WebQSP"])
    parser.add_argument("--cwq-trace", type=Path, default=DEFAULT_TRACE_FILES["CWQ"])
    parser.add_argument("--shuffle-seed", type=int, default=0, help="seed for deterministic shuffled access")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="base seed for the 'random' eviction policy")
    parser.add_argument("--random-repeats", type=int, default=DEFAULT_RANDOM_REPEATS,
                        help="number of seeds to average the 'random' policy over")
    args = parser.parse_args()

    cache_sizes = _parse_int_list(args.cache_sizes)
    policies = _parse_str_list(args.policies)
    if args.combined:
        payload = build_combined_summary(
            trace_files={
                "WebQSP": args.webqsp_trace,
                "CWQ": args.cwq_trace,
            },
            cache_sizes=cache_sizes,
            policies=policies,
            shuffle_seed=args.shuffle_seed,
            random_seed=args.random_seed,
            random_repeats=args.random_repeats,
        )
        output_path = args.output or DEFAULT_COMBINED_OUTPUT
    else:
        if not args.trace_path:
            parser.error("trace_path is required unless --combined is set")
        traces = load_traces(args.trace_path)
        results = run_simulation(
            traces=traces,
            cache_sizes=cache_sizes,
            policies=policies,
            random_seed=args.random_seed,
            random_repeats=args.random_repeats,
        )
        payload = [result.to_dict() for result in results]
        output_path = args.output

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, indent=2)
            outfile.write("\n")
        print(f"Saved: {output_path}")
        return

    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
