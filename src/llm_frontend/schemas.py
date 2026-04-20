from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_ACTIONS = frozenset({"INITIAL_ENTITY", "KG_QUERY", "FINAL_ANSWER"})
VALID_DIRECTIONS = frozenset({"auto", "forward", "backward"})


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def normalize_direction(value: Any) -> str:
    text = str(value or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "forward": "forward",
        "backward": "backward",
        "out": "forward",
        "in": "backward",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported direction: {value}")
    return aliases[text]


@dataclass(frozen=True)
class QuestionExample:
    question_id: str
    question: str
    gold_inferential_chain: list[str] = field(default_factory=list)
    gold_answers: list[str] = field(default_factory=list)
    split: str | None = None


@dataclass(frozen=True)
class InitialEntityAction:
    action: str
    entity: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "entity": self.entity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KGQueryAction:
    action: str
    relation: str
    direction: str = "auto"
    frontier: str = "CURRENT_FRONTIER"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "relation": self.relation,
            "direction": self.direction,
            "frontier": self.frontier,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FinalAnswerAction:
    action: str
    answers: list[str]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "answers": self.answers,
            "reason": self.reason,
        }


PlannerAction = InitialEntityAction | KGQueryAction | FinalAnswerAction


@dataclass(frozen=True)
class FrontierObservation:
    frontier: list[str]
    frontier_size: int
    sample_entities: list[str]
    forward_relations: list[str]
    backward_relations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontier": self.frontier,
            "frontier_size": self.frontier_size,
            "sample_entities": self.sample_entities,
            "forward_relations": self.forward_relations,
            "backward_relations": self.backward_relations,
        }


@dataclass(frozen=True)
class BackendQueryResult:
    resolved_direction: str
    output_frontier: list[str]
    observation: FrontierObservation


@dataclass(frozen=True)
class LLMQueryTraceStep:
    step_id: int
    relation: str
    direction: str
    resolved_direction: str
    output_frontier: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "relation": self.relation,
            "direction": self.direction,
            "resolved_direction": self.resolved_direction,
            "output_frontier": self.output_frontier,
        }


@dataclass(frozen=True)
class LLMRunTrace:
    question_id: str
    question: str
    llm_initial_entity: str | None
    llm_initial_frontier: list[str]
    llm_kg_queries: list[LLMQueryTraceStep]
    llm_final_answer: list[str]
    num_steps: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "llm_initial_entity": self.llm_initial_entity,
            "llm_initial_frontier": self.llm_initial_frontier,
            "llm_kg_queries": [step.to_dict() for step in self.llm_kg_queries],
            "llm_final_answer": self.llm_final_answer,
            "num_steps": self.num_steps,
            "stop_reason": self.stop_reason,
        }


def parse_planner_action(payload: dict[str, Any]) -> PlannerAction:
    action = str(payload.get("action", "")).strip().upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported action: {action or '<missing>'}")

    if action == "INITIAL_ENTITY":
        entity = str(payload.get("entity", "")).strip()
        if not entity:
            raise ValueError("INITIAL_ENTITY requires a non-empty entity")
        return InitialEntityAction(
            action="INITIAL_ENTITY",
            entity=entity,
            reason=str(payload.get("reason", "")).strip() or None,
        )

    if action == "KG_QUERY":
        relation = str(payload.get("relation", "")).strip()
        if not relation:
            raise ValueError("KG_QUERY requires a non-empty relation")
        frontier = (
            str(payload.get("frontier", "CURRENT_FRONTIER")).strip()
            or "CURRENT_FRONTIER"
        )
        if frontier != "CURRENT_FRONTIER":
            raise ValueError("Only frontier='CURRENT_FRONTIER' is supported")
        return KGQueryAction(
            action="KG_QUERY",
            relation=relation,
            direction=normalize_direction(payload.get("direction", "auto")),
            frontier="CURRENT_FRONTIER",
            reason=str(payload.get("reason", "")).strip() or None,
        )

    answers_value = payload.get("answers") or []
    if isinstance(answers_value, str):
        answers_value = [answers_value]
    if not isinstance(answers_value, list):
        raise ValueError("FINAL_ANSWER requires a list of answers")
    return FinalAnswerAction(
        action="FINAL_ANSWER",
        answers=unique_strings([str(answer) for answer in answers_value]),
        reason=str(payload.get("reason", "")).strip() or None,
    )
