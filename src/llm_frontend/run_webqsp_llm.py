from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_cache_backend import load_webqsp_examples
from kg_cache_backend.webqsp_loader import resolve_webqsp_path

from .backend_adapter import KGBackendAdapter
from .config import LLMFrontendConfig
from .controller import IterativeKGController
from .llm_client import LLMChatClient
from .planner import LLMPlanner
from .schemas import QuestionExample
from .trace import summarize_traces, write_trace_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the iterative LLM frontend on WebQSP or one question.")
    parser.add_argument("--triples", type=Path, required=True, help="Flat triple JSONL or local subgraph file.")
    parser.add_argument("--question", type=str, default=None, help="Run one ad-hoc question.")
    parser.add_argument("--question-id", type=str, default="manual-0")
    parser.add_argument("--topic-entity", type=str, default=None)
    parser.add_argument("--webqsp", type=Path, default=None, help="WebQSP file or dataset directory.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL for LLM traces.")
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "gemini"],
        default=None,
        help="Override the LLM provider. Defaults to LLM_PROVIDER or gemini.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override the LLM model name.")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache", choices=["none", "lru", "lfu"], default="none")
    parser.add_argument("--cache-capacity", type=int, default=0)
    return parser


def _question_example_from_trace(trace) -> QuestionExample:
    return QuestionExample(
        question_id=trace.question_id,
        question=trace.raw_question,
        topic_entity=trace.topic_entity,
        gold_inferential_chain=list(trace.inferential_chain),
        gold_answers=trace.gold_answer_ids(),
        split=trace.split,
    )


def _load_examples(args: argparse.Namespace) -> list[QuestionExample]:
    if args.question:
        return [
            QuestionExample(
                question_id=args.question_id,
                question=args.question,
                topic_entity=args.topic_entity,
                split=None,
            )
        ]
    if args.webqsp is None:
        raise SystemExit("Provide either --question or --webqsp.")
    resolved_webqsp = resolve_webqsp_path(args.webqsp, args.split)
    traces = load_webqsp_examples(resolved_webqsp, split=args.split, limit=args.limit)
    return [_question_example_from_trace(trace) for trace in traces]


def main() -> None:
    args = build_parser().parse_args()
    if args.question and args.webqsp is not None:
        raise SystemExit("Use either --question or --webqsp, not both.")

    default_config = LLMFrontendConfig()
    config = LLMFrontendConfig(
        provider=args.provider or default_config.provider,
        model=args.model,
        temperature=args.temperature,
        max_steps=args.max_steps,
    )
    client = LLMChatClient.from_config(
        config,
        model=args.model,
        provider=args.provider,
    )
    planner = LLMPlanner(client=client, config=config)
    backend = KGBackendAdapter.from_path(
        triples_path=args.triples,
        config=config,
        cache_mode=args.cache,
        cache_capacity=args.cache_capacity,
    )
    controller = IterativeKGController(planner=planner, backend=backend, config=config)

    examples = _load_examples(args)
    traces = [controller.run(example) for example in examples]

    if args.output is not None:
        write_trace_jsonl(args.output, traces)

    summary = summarize_traces(traces)
    summary.update(
        {
            "provider": client.provider,
            "triples_path": str(args.triples),
            "output_path": str(args.output) if args.output is not None else None,
            "model": client.model,
            "base_url": client.base_url,
            "cache_mode": backend.cache.mode,
            "cache_capacity": backend.cache.capacity,
            "cache_hits": backend.cache.hits,
            "cache_misses": backend.cache.misses,
            "primitive_calls": backend.cache.primitive_calls,
            "store_triples": backend.store.triple_count,
            "store_entities": backend.store.entity_count,
            "store_relations": backend.store.relation_count,
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
