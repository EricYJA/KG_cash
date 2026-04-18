from __future__ import annotations

from dataclasses import dataclass
import json

from .config import LLMFrontendConfig
from .llm_client import ChatMessage, LLMClient
from .memory import PlannerMemory
from .prompts import build_system_prompt, build_user_prompt
from .schemas import FrontierObservation, PlannerAction, QuestionExample, parse_planner_action


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
    action: PlannerAction | None
    raw_output: str
    error: str | None = None


class LLMPlanner:
    def __init__(self, client: LLMClient, config: LLMFrontendConfig) -> None:
        self.client = client
        self.config = config

    def plan_next(
        self,
        example: QuestionExample,
        memory: PlannerMemory,
        observation: FrontierObservation,
    ) -> PlannerDecision:
        messages = [
            ChatMessage(role="system", content=build_system_prompt(self.config)),
            ChatMessage(
                role="user",
                content=build_user_prompt(example=example, memory=memory, observation=observation),
            ),
        ]
        raw_output = self.client.complete_json(messages, temperature=self.config.temperature)
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
            action = parse_planner_action(payload)
        except ValueError as exc:
            return PlannerDecision(action=None, raw_output=raw_output, error=str(exc))
        return PlannerDecision(action=action, raw_output=raw_output)
