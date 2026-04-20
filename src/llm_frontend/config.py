from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMFrontendConfig:
    """Controller settings for the iterative LLM frontend."""

    temperature: float = 0.0
    initial_entity_search_limit: int = 3
    max_steps: int = 6
    max_memory_steps: int = 6
    max_frontier_entities: int = 8
    max_relation_candidates: int = 12
    relation_scan_limit: int = 12
    repeat_query_limit: int = 2
    repeat_frontier_limit: int = 3
    fallback_answer_limit: int = 10
    request_timeout_s: float = 60.0
