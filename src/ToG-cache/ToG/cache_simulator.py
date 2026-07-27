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
SUPPORTED_POLICIES = ("lru", "lfu", "oracle")
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


def _simulate_lru(requests: list[KGRequest], cache_size: int) -> tuple[int, int]:
    if cache_size <= 0:
        return 0, len(requests)
    cache: OrderedDict[str, None] = OrderedDict()
    hits = 0
    for request in requests:
        if request.key in cache:
            cache.move_to_end(request.key)
            hits += 1
        else:
            if len(cache) >= cache_size:
                cache.popitem(last=False)
            cache[request.key] = None
    return hits, len(requests) - hits


def _simulate_lfu(requests: list[KGRequest], cache_size: int) -> tuple[int, int]:
    if cache_size <= 0:
        return 0, len(requests)
    cache: dict[str, None] = {}
    freq: Counter[str] = Counter()
    hits = 0
    for request in requests:
        if request.key in cache:
            freq[request.key] += 1
            hits += 1
        else:
            if len(cache) >= cache_size:
                lfu_key = min(freq, key=lambda key: freq[key])
                del cache[lfu_key]
                del freq[lfu_key]
            cache[request.key] = None
            freq[request.key] = 1
    return hits, len(requests) - hits


def _simulate_oracle(requests: list[KGRequest], cache_size: int) -> tuple[int, int]:
    if cache_size <= 0:
        return 0, len(requests)
    freq = Counter(request.key for request in requests)
    preloaded = {key for key, _ in freq.most_common(cache_size)}
    hits = sum(1 for request in requests if request.key in preloaded)
    return hits, len(requests) - hits


def _simulate_kg_time(requests: list[KGRequest], policy: str, cache_size: int) -> tuple[int, int, int]:
    if policy == "lru":
        hits, misses = _simulate_lru(requests, cache_size)
    elif policy == "lfu":
        hits, misses = _simulate_lfu(requests, cache_size)
    elif policy == "oracle":
        hits, misses = _simulate_oracle(requests, cache_size)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    kg_simulated_ms = 0
    if policy == "oracle":
        freq = Counter(request.key for request in requests)
        cached = {key for key, _ in freq.most_common(max(cache_size, 0))}
        for request in requests:
            if request.key not in cached:
                kg_simulated_ms += request.duration_ms
    elif policy == "lru":
        if cache_size <= 0:
            kg_simulated_ms = sum(request.duration_ms for request in requests)
        else:
            cache: OrderedDict[str, None] = OrderedDict()
            for request in requests:
                if request.key in cache:
                    cache.move_to_end(request.key)
                    continue
                kg_simulated_ms += request.duration_ms
                if len(cache) >= cache_size:
                    cache.popitem(last=False)
                cache[request.key] = None
    else:
        if cache_size <= 0:
            kg_simulated_ms = sum(request.duration_ms for request in requests)
        else:
            cache: dict[str, None] = {}
            freq: Counter[str] = Counter()
            for request in requests:
                if request.key in cache:
                    freq[request.key] += 1
                    continue
                kg_simulated_ms += request.duration_ms
                if len(cache) >= cache_size:
                    lfu_key = min(freq, key=lambda key: freq[key])
                    del cache[lfu_key]
                    del freq[lfu_key]
                cache[request.key] = None
                freq[request.key] = 1

    return hits, misses, kg_simulated_ms


def run_simulation(
    traces: list[dict],
    cache_sizes: list[int],
    policies: list[str],
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
    )


def run_simulation_from_requests(
    requests: list[KGRequest],
    breakdown: dict[str, int],
    cache_sizes: list[int],
    policies: list[str],
) -> list[CacheSimResult]:
    invalid_policies = [policy for policy in policies if policy not in SUPPORTED_POLICIES]
    if invalid_policies:
        raise ValueError(f"Unsupported policies: {', '.join(invalid_policies)}")

    results: list[CacheSimResult] = []
    for size in cache_sizes:
        for policy in policies:
            hits, misses, kg_simulated_ms = _simulate_kg_time(requests, policy, size)
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
) -> dict[str, object]:
    payload: dict[str, object] = {
        "metadata": {
            "cache_sizes": cache_sizes,
            "policies": policies,
            "shuffle_seed": shuffle_seed,
            "access_patterns": ["sequential", "request_block_shuffled"],
            "shuffled_semantics": "Question/request traces are shuffled; internal KG access order is preserved.",
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
                )
            ],
            "shuffled": [
                result.to_dict()
                for result in run_simulation_from_requests(
                    requests=shuffled_requests,
                    breakdown=breakdown,
                    cache_sizes=cache_sizes,
                    policies=policies,
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
    parser.add_argument("--policies", default="lru,lfu,oracle", help="comma-separated policies")
    parser.add_argument("--pretty", action="store_true", help="pretty print the simulation results as JSON")
    parser.add_argument("--output", type=Path, help="write simulation JSON to this path")
    parser.add_argument("--combined", action="store_true", help="simulate WebQSP and CWQ into one JSON file")
    parser.add_argument("--webqsp-trace", type=Path, default=DEFAULT_TRACE_FILES["WebQSP"])
    parser.add_argument("--cwq-trace", type=Path, default=DEFAULT_TRACE_FILES["CWQ"])
    parser.add_argument("--shuffle-seed", type=int, default=0, help="seed for deterministic shuffled access")
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
