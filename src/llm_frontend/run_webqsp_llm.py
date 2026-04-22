from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kg_backend.backend import UncachedKGBackend

from .backend_adapter import KGBackendAdapter
from .config import LLMFrontendConfig
from .controller import IterativeKGController
from .controller_direct import DirectKGController
from .llm_config import (
    DEFAULT_LLM_VENDOR,
    LLM_API_KEY_ENV,
    LLM_VENDOR_CHOICES,
    resolve_llm_config,
)
from .llm_client import LLMChatClient
from .planner import LLMPlanner
from .schemas import QuestionExample
from .trace import summarize_traces, write_trace_jsonl
from .webqsp_loader import load_webqsp_examples



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the iterative LLM frontend on WebQSP or one question."
    )
    parser.add_argument(
        "--kg-path",
        "--triples",
        dest="kg_path",
        type=Path,
        default=Path("datasets/WebQSP_KG"),
        help="KG directory or triples file for kg_backend.",
    )
    parser.add_argument(
        "--question", type=str, default=None, help="Run one ad-hoc question."
    )
    parser.add_argument("--question-id", type=str, default="manual-0")
    parser.add_argument(
        "--webqsp", type=Path, default="datasets/WebQSP", help="WebQSP file or dataset directory."
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        # default=["WebQTrn-1"],
        default=None,
        dest="question_ids",
        help="Run only these specific question IDs (e.g. WebQTrn-0 WebQTrn-5).",
    )
    parser.add_argument(
        "--vendor",
        "--VENDOR",
        dest="vendor",
        choices=LLM_VENDOR_CHOICES,
        default=DEFAULT_LLM_VENDOR,
        help="Vendor preset from llm_config.py.",
    )
    parser.add_argument(
        "--api-key",
        "--API_KEY",
        dest="api_key",
        type=str,
        default=None,
        help=f"Override the API key instead of using {LLM_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--base-url",
        "--BASE_URL",
        dest="base_url",
        type=str,
        default=None,
        help="Override the preset base URL.",
    )
    parser.add_argument(
        "--model",
        "--MODEL",
        dest="model",
        type=str,
        default=None,
        help="Override the preset model name.",
    )
    parser.add_argument(
        "--initial-entity-search-limit",
        type=int,
        default=3,
        help="Maximum number of LLM initial-entity attempts before returning an empty answer.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--controller",
        choices=["iterative", "direct"],
        default="iterative",
        help="iterative=two-phase (relation then entity); direct=single-phase (entity from neighborhood).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/traces_iterative.jsonl"), help="Output JSONL for LLM traces."
    )
    return parser


def _load_examples(args: argparse.Namespace) -> list[QuestionExample]:
    if args.question:
        return [
            QuestionExample(
                question_id=args.question_id,
                question=args.question,
                split=None,
            )
        ]
    if args.webqsp is None:
        raise SystemExit("Provide either --question or --webqsp.")
    examples = load_webqsp_examples(args.webqsp, split=args.split, limit=args.limit)
    if args.question_ids:
        ids = set(args.question_ids)
        examples = [e for e in examples if e.question_id in ids]
    return examples


def main() -> None:
    args = build_parser().parse_args()
    if args.question and args.webqsp is not None:
        raise SystemExit("Use either --question or --webqsp, not both.")

    connection_config = resolve_llm_config(
        vendor=args.vendor,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
    )
    config = LLMFrontendConfig(
        temperature=args.temperature,
        initial_entity_search_limit=args.initial_entity_search_limit,
        max_steps=args.max_steps,
    )
    client = LLMChatClient.from_connection_config(
        connection_config=connection_config,
        timeout_s=config.request_timeout_s,
    )
    planner = LLMPlanner(client=client, config=config)
    backend = KGBackendAdapter(
        backend=UncachedKGBackend.from_data_path(args.kg_path),
        config=config,
    )
    if args.controller == "direct":
        controller = DirectKGController(planner=planner, backend=backend, config=config)
    else:
        controller = IterativeKGController(planner=planner, backend=backend, config=config)

    examples = _load_examples(args)
    traces = []
    for i, example in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] Running {example.question_id}: {example.question}")
        trace = controller.run(example)
        traces.append(trace)
        has_answer = bool(trace.llm_final_answer)
        hit = (
            bool(set(trace.llm_final_answer) & set(trace.gold_answers))
            if trace.gold_answers else None
        )
        hit_str = f"  hit1={'YES' if hit else 'NO'}" if hit is not None else ""
        answer_str = f"  answer={trace.llm_final_answer}" if has_answer else "  no answer"
        print(f"         stop={trace.stop_reason}{answer_str}{hit_str}")

    write_trace_jsonl(args.output, traces)

    summary = summarize_traces(traces)
    stats = backend.stats()
    summary.update(
        {
            "api": client.api_name,
            "vendor": client.vendor,
            "kg_path": str(args.kg_path),
            "output_path": str(args.output),
            "model": client.model,
            "base_url": client.base_url,
            "api_base_url": client.api_base_url,
            "num_entities": stats.num_entities,
            "num_relations": stats.num_relations,
            "num_triples": stats.num_triples,
        }
    )
    # print(json.dumps(summary, indent=2))
    if "hit1" in summary:
        print(f"Hit@1: {summary['hit1']:.2%}")


if __name__ == "__main__":
    main()
