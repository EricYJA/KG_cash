from __future__ import annotations

from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .schemas import FrontierObservation, QuestionExample


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def build_system_prompt(config: LLMFrontendConfig) -> str:
    return (
        "Return exactly one JSON object and nothing else. "
        'Allowed outputs: {"action":"INITIAL_ENTITY","entity":"...","reason":"short"} '
        'or {"action":"KG_QUERY","relation":"...","direction":"auto|forward|backward",'
        '"frontier":"CURRENT_FRONTIER","reason":"short"} '
        'or {"action":"FINAL_ANSWER","answers":["..."],"reason":"short"}. '
        "Use INITIAL_ENTITY when there is no current frontier yet. "
        "INITIAL_ENTITY should propose a human-readable entity name or an exact KG id. "
        "Use KG_QUERY for one more KG hop. Use FINAL_ANSWER when the current frontier is enough. "
        "Only frontier CURRENT_FRONTIER is allowed. Keep reason short. "
        f"At most {config.initial_entity_search_limit} initial-entity attempts are allowed. "
        f"At most {config.max_steps} KG queries are allowed."
    )


def build_user_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
    observation: FrontierObservation,
) -> str:
    if observation.frontier_size == 0:
        return (
            f"Question: {example.question}\n"
            "Current frontier: (empty)\n"
            f"Failed initial entity attempts: {memory.format_failed_initial_entities()}\n"
            f"History: {memory.format_history()}\n"
            "Choose INITIAL_ENTITY now using a human-readable entity name or an exact KG id. "
            "If a prior attempt failed, choose a different or more precise name."
        )

    return (
        f"Question: {example.question}\n"
        f"Frontier size: {observation.frontier_size}\n"
        f"Frontier sample: {_format_list(observation.sample_entities)}\n"
        f"Forward relations: {_format_list(observation.forward_relations)}\n"
        f"Backward relations: {_format_list(observation.backward_relations)}\n"
        f"History: {memory.format_history()}\n"
        "Choose the next action now."
    )


def build_compact_user_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
    observation: FrontierObservation,
) -> str:
    """Build a minimal fallback planner prompt for brittle chat endpoints."""

    forward_relations = observation.forward_relations[:6]
    backward_relations = observation.backward_relations[:6]
    history = memory.format_history()
    if history == "No prior KG queries.":
        history = "none"

    if observation.frontier_size == 0:
        return (
            "JSON only. "
            'Return {"action":"INITIAL_ENTITY","entity":"name_or_id","reason":"short"}. '
            f"Failed={memory.format_failed_initial_entities()} "
            f"Q={example.question} "
            f"Hist={history}"
        )

    return (
        "JSON only. "
        'Either return {"action":"KG_QUERY","relation":"REL","direction":"auto|forward|backward",'
        '"frontier":"CURRENT_FRONTIER","reason":"short"} '
        'or {"action":"FINAL_ANSWER","answers":["ID"],"reason":"short"}. '
        f"Q={example.question} "
        f"Frontier={_format_list(observation.sample_entities[:4])} "
        f"Fwd={_format_list(forward_relations)} "
        f"Back={_format_list(backward_relations)} "
        f"Hist={history}"
    )
