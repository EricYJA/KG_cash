from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache_simulator import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate cache policies over saved LLM trace JSONL."
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
    )
    parser.add_argument(
        "--cache-sizes",
        nargs="+",
        type=int,
        default=[10, 50, 100, 500, 1000],
        dest="cache_sizes",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle questions before simulation.",
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
        default="iterative_cache_summary_no_shuffle.json",
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

    results = run_simulation(traces, args.cache_sizes, args.policies, shuffle=args.shuffle, seed=args.seed)

    print(f"\n{'Policy':<8} {'Size':>6} {'Requests':>10} {'Hits':>8} {'Misses':>8} {'HitRate':>8}")
    print("-" * 54)
    for r in results:
        print(
            f"{r.policy:<8} {r.cache_size:>6} {r.requests:>10} "
            f"{r.hits:>8} {r.misses:>8} {r.hit_rate:>8.2%}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
