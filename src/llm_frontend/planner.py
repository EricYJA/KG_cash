from __future__ import annotations

from dataclasses import dataclass
import json

from .config import LLMFrontendConfig
from .llm_client import ChatMessage, LLMClient
from .memory import PlannerMemory
from .prompts import (
    build_direct_prompt,
    build_direct_system_prompt,
    build_entity_evaluation_prompt,
    build_entity_evaluation_system_prompt,
    build_initial_entity_prompt,
    build_initial_entity_system_prompt,
    build_relation_selection_prompt,
    build_relation_selection_system_prompt,
)

from .schemas import (
    ExploreAction,
    ExploreMultiAction,
    FinalAnswerAction,
    FrontierObservation,
    InitialEntityAction,
    KGQueryAction,
    QuestionExample,
    parse_direct_action,
    parse_entity_evaluation_action,
    parse_initial_entity_action,
    parse_relation_selection_action,
)

PhasedAction = InitialEntityAction | KGQueryAction | FinalAnswerAction | ExploreAction


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class PlannerDecision:
    action: PhasedAction | None
    raw_output: str
    error: str | None = None


class LLMPlanner:
    def __init__(self, client: LLMClient, config: LLMFrontendConfig) -> None:
        self.client = client
        self.config = config

    def select_initial_entity(
        self,
        example: QuestionExample,
        memory: PlannerMemory,
    ) -> PlannerDecision:
        messages = [
            ChatMessage(
                role="system",
                content=build_initial_entity_system_prompt(self.config),
            ),
            ChatMessage(
                role="user",
                content=build_initial_entity_prompt(example, memory),
            ),
        ]
        raw_output = self._call(messages)
        if isinstance(raw_output, RuntimeError):
            return PlannerDecision(action=None, raw_output=str(raw_output), error=str(raw_output))
        return self._parse(raw_output, parse_initial_entity_action)

    def select_relation(
        self,
        example: QuestionExample,
        memory: PlannerMemory,
        frontier: list[str],
        frontier_labels: dict[str, str],
        observation: FrontierObservation,
    ) -> PlannerDecision:
        messages = [
            ChatMessage(
                role="system",
                content=build_relation_selection_system_prompt(),
            ),
            ChatMessage(
                role="user",
                content=build_relation_selection_prompt(
                    example, memory, frontier, frontier_labels, observation
                ),
            ),
        ]
        raw_output = self._call(messages)
        if isinstance(raw_output, RuntimeError):
            return PlannerDecision(action=None, raw_output=str(raw_output), error=str(raw_output))
        return self._parse(raw_output, parse_relation_selection_action)

    def evaluate_entities(
        self,
        example: QuestionExample,
        memory: PlannerMemory,
        destination_entities: list[str],
        destination_labels: dict[str, str],
    ) -> PlannerDecision:
        messages = [
            ChatMessage(
                role="system",
                content=build_entity_evaluation_system_prompt(self.config),
            ),
            ChatMessage(
                role="user",
                content=build_entity_evaluation_prompt(
                    example, memory, destination_entities, destination_labels
                ),
            ),
        ]
        raw_output = self._call(messages)
        if isinstance(raw_output, RuntimeError):
            return PlannerDecision(action=None, raw_output=str(raw_output), error=str(raw_output))
        return self._parse(raw_output, parse_entity_evaluation_action)

    def select_next_entity(
        self,
        example: QuestionExample,
        memory: PlannerMemory,
        frontier: list[str],
        frontier_labels: dict[str, str],
        neighborhood: list[tuple[str, list[str]]],
        neighbor_labels: dict[str, str],
    ) -> PlannerDecision:
        messages = [
            ChatMessage(role="system", content=build_direct_system_prompt(self.config.max_explore_entities)),
            ChatMessage(
                role="user",
                content=build_direct_prompt(
                    example, memory, frontier, frontier_labels, neighborhood, neighbor_labels
                ),
            ),
        ]
        raw_output = self._call(messages)
        if isinstance(raw_output, RuntimeError):
            return PlannerDecision(action=None, raw_output=str(raw_output), error=str(raw_output))
        return self._parse(raw_output, parse_direct_action)

    def _call(self, messages: list[ChatMessage]) -> str | RuntimeError:
        try:
            return self.client.complete_json(messages, temperature=self.config.temperature)
        except RuntimeError as exc:
            return exc

    def _parse(self, raw_output: str, parser) -> PlannerDecision:
        cleaned = _strip_markdown_fences(raw_output)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return PlannerDecision(
                action=None,
                raw_output=raw_output,
                error=f"Invalid JSON output: {exc.msg}",
            )
        try:
            action = parser(payload)
        except ValueError as exc:
            return PlannerDecision(action=None, raw_output=raw_output, error=str(exc))
        return PlannerDecision(action=action, raw_output=raw_output)

