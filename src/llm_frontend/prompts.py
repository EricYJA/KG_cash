from __future__ import annotations

from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .schemas import FrontierObservation, QuestionExample


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _format_entities_with_labels(
    entity_ids: list[str], labels: dict[str, str]
) -> str:
    parts = []
    for eid in entity_ids:
        label = labels.get(eid)
        parts.append(f"{eid} ({label})" if label else eid)
    return ", ".join(parts) if parts else "(none)"


# ---------------------------------------------------------------------------
# Initial entity phase
# ---------------------------------------------------------------------------

def build_initial_entity_system_prompt(config: LLMFrontendConfig) -> str:
    return (
        "Return exactly one JSON object and nothing else. "
        'Return: {"action":"INITIAL_ENTITY","entity":"...","reason":"..."} '
        "Propose a human-readable entity name or an exact KG id to start the search. "
        "reason: one short phrase explaining why this entity is the main topic of the question."
    )


def build_initial_entity_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
) -> str:
    prompt = f"Question: {example.question}\n"
    if memory.failed_initial_entities:
        failed = ", ".join(memory.failed_initial_entities)
        prompt += (
            f"Previously tried entities not found in the KG: {failed}\n"
            "Try a different spelling, a more common name, or a related entity.\n"
        )
    prompt += "Identify the main topic entity in the question."
    return prompt


# ---------------------------------------------------------------------------
# Phase 1: Relation selection
# ---------------------------------------------------------------------------

def build_relation_selection_system_prompt() -> str:
    return (
        "Return exactly one JSON object and nothing else. "
        'Return: {"action":"KG_QUERY","relation":"...","direction":"auto|forward|backward","reason":"..."} '
        "Pick the single most relevant relation to follow to answer the question. "
        "You MUST use a relation name exactly as it appears in the forward or backward relations list. Do not invent or modify relation names. "
        'direction "auto" tries forward first then backward. '
        "reason: one short phrase explaining why this relation leads toward the answer."
    )


def build_relation_selection_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
    frontier: list[str],
    frontier_labels: dict[str, str],
    observation: FrontierObservation,
) -> str:
    return (
        f"Question: {example.question}\n"
        f"Current frontier ({observation.frontier_size} entities): "
        f"{_format_entities_with_labels(observation.sample_entities, frontier_labels)}\n"
        f"Forward relations: {_format_list(observation.forward_relations)}\n"
        f"Backward relations: {_format_list(observation.backward_relations)}\n"
        f"History: {memory.format_history()}\n"
        "Pick the best relation to follow next."
    )


# ---------------------------------------------------------------------------
# Phase 2: Entity evaluation
# ---------------------------------------------------------------------------

def build_entity_evaluation_system_prompt(config: LLMFrontendConfig) -> str:
    return (
        "Return exactly one JSON object and nothing else. "
        'If the listed entities answer the question, return: {"action":"FINAL_ANSWER","answers":["entity_id"],"reason":"..."} '
        "Use entity IDs from the list as answers. "
        "reason: one short phrase explaining why these entities answer the question. "
        'If these entities do not yet answer the question, return: {"action":"EXPLORE","entity":"entity_id","reason":"..."} '
        "Pick the single most promising entity ID from the list to continue exploring. "
        "reason: one short phrase explaining why this entity is worth exploring further. "
        "You MUST use entity IDs exactly as they appear in the destination entities list. Do not invent or modify entity IDs."
    )


def build_entity_evaluation_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
    destination_entities: list[str],
    destination_labels: dict[str, str],
) -> str:
    last_relation = memory.turns[-1].relation if memory.turns else "unknown"
    return (
        f"Question: {example.question}\n"
        f'After following relation "{last_relation}", the destination entities are:\n'
        f"{_format_entities_with_labels(destination_entities, destination_labels)}\n"
        f"History: {memory.format_history()}\n"
        "Do these entities answer the question? "
        "Return FINAL_ANSWER with the answer entity IDs, or EXPLORE with the top most promising entity ID to keep searching."
    )


# ---------------------------------------------------------------------------
# Direct controller: single-phase (relation + entity combined)
# ---------------------------------------------------------------------------

def build_direct_system_prompt(max_explore: int = 3) -> str:
    return (
        "Return exactly one JSON object and nothing else. "
        'If any of the listed entities answer the question, return: {"action":"FINAL_ANSWER","answers":["entity_id1",...],"reason":"..."} '
        f'Otherwise return: {{"action":"EXPLORE","entities":["entity_id1",...],"reason":"..."}} '
        f"Include up to {max_explore} most promising entity IDs — pick more than one if genuinely useful, but no more than {max_explore}. "
        "You MUST use entity IDs exactly as they appear in the neighborhood list. Do not invent or modify entity IDs. "
        "reason: one short phrase."
    )


def build_direct_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
    frontier: list[str],
    frontier_labels: dict[str, str],
    neighborhood: list[tuple[str, list[str]]],
    neighbor_labels: dict[str, str],
) -> str:
    lines = [
        f"Question: {example.question}",
        f"Current entities: {_format_entities_with_labels(frontier, frontier_labels)}",
        "Neighborhood (relation → destination entities):",
    ]
    for relation, neighbors in neighborhood:
        neighbor_str = _format_entities_with_labels(neighbors, neighbor_labels)
        lines.append(f"  {relation} → {neighbor_str}")
    lines.append(f"History: {memory.format_history()}")
    lines.append("Pick the best next entity to explore, or give the final answer.")
    return "\n".join(lines)
