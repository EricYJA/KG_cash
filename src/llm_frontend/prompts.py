from __future__ import annotations

from .config import LLMFrontendConfig
from .memory import PlannerMemory
from .schemas import FrontierObservation, QuestionExample


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def build_system_prompt(config: LLMFrontendConfig) -> str:
    return (
        "You are an iterative knowledge-graph planner.\n"
        "You interact with a backend one step at a time.\n"
        "The backend starts from the current frontier, follows exactly one relation hop, "
        "and returns the updated frontier plus a short frontier summary.\n"
        "You will also see candidate forward and backward relations from the current frontier.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Allowed actions:\n"
        '1. {"action":"KG_QUERY","relation":"...","direction":"forward","entity":null,'
        '"frontier":"CURRENT_FRONTIER","reason":"short note"}\n'
        '2. {"action":"FINAL_ANSWER","answers":["..."],"reason":"short note"}\n'
        "Rules:\n"
        "- Use KG_QUERY only when another KG hop is needed.\n"
        "- Use FINAL_ANSWER when the current evidence is sufficient.\n"
        "- If the current frontier view is complete and already answers the question, "
        "copy those entity ids into FINAL_ANSWER.answers.\n"
        "- Keep reason short. Do not output hidden reasoning or long explanations.\n"
        "- Only frontier='CURRENT_FRONTIER' is supported.\n"
        "- direction is your best guess. The backend may resolve the actual hop direction automatically.\n"
        f"- The controller will stop after at most {config.max_steps} KG queries.\n"
    )


def build_user_prompt(
    example: QuestionExample,
    memory: PlannerMemory,
    observation: FrontierObservation,
) -> str:
    frontier_view = (
        "complete"
        if observation.frontier_size == len(observation.sample_entities)
        else "truncated"
    )
    return (
        f"Question: {example.question}\n"
        f"Question ID: {example.question_id}\n"
        f"Topic entity: {example.topic_entity or '(unknown)'}\n"
        f"Steps used: {memory.steps_used}\n"
        f"Steps remaining: {memory.steps_remaining}\n"
        f"Current frontier size: {observation.frontier_size}\n"
        f"Current frontier view: {frontier_view}\n"
        f"Current frontier entities shown: {_format_list(observation.sample_entities)}\n"
        f"Candidate forward relations: {_format_list(observation.forward_relations)}\n"
        f"Candidate backward relations: {_format_list(observation.backward_relations)}\n"
        "Recent KG history:\n"
        f"{memory.format_history()}\n"
        "Decide the next action now."
    )
