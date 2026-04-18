from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_ACTIONS = frozenset({"KG_QUERY", "FINAL_ANSWER"})
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
    topic_entity: str | None = None
    gold_inferential_chain: list[str] = field(default_factory=list)
    gold_answers: list[str] = field(default_factory=list)
    split: str | None = None


@dataclass(frozen=True)
class KGQueryAction:
    action: str
    relation: str
    direction: str = "auto"
    entity: str | None = None
    frontier: str = "CURRENT_FRONTIER"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "relation": self.relation,
            "direction": self.direction,
            "entity": self.entity,
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


PlannerAction = KGQueryAction | FinalAnswerAction


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
    relation: str
    requested_direction: str
    resolved_direction: str
    input_frontier: list[str]
    frontier_after_hop: list[str]
    output_frontier: list[str]
    primitive_calls: int
    cache_hits: int
    cache_misses: int
    observation: FrontierObservation


@dataclass(frozen=True)
class LLMQueryTraceStep:
    step_id: int
    relation: str
    direction: str
    requested_direction: str
    input_frontier: list[str]
    output_frontier: list[str]
    raw_model_output: str
    reason: str | None = None
    frontier_after_hop: list[str] = field(default_factory=list)
    primitive_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "relation": self.relation,
            "direction": self.direction,
            "requested_direction": self.requested_direction,
            "input_frontier": self.input_frontier,
            "output_frontier": self.output_frontier,
            "raw_model_output": self.raw_model_output,
            "reason": self.reason,
            "frontier_after_hop": self.frontier_after_hop,
            "primitive_calls": self.primitive_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass(frozen=True)
class LLMRunTrace:
    question_id: str
    question: str
    topic_entity: str | None
    gold_inferential_chain: list[str]
    llm_kg_queries: list[LLMQueryTraceStep]
    llm_final_answer: list[str]
    gold_answers: list[str]
    num_steps: int
    stop_reason: str
    llm_final_reason: str | None = None
    final_raw_model_output: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "topic_entity": self.topic_entity,
            "gold_inferential_chain": self.gold_inferential_chain,
            "llm_kg_queries": [step.to_dict() for step in self.llm_kg_queries],
            "llm_final_answer": self.llm_final_answer,
            "gold_answers": self.gold_answers,
            "num_steps": self.num_steps,
            "stop_reason": self.stop_reason,
            "llm_final_reason": self.llm_final_reason,
            "final_raw_model_output": self.final_raw_model_output,
            "split": self.split,
        }


def parse_planner_action(payload: dict[str, Any]) -> PlannerAction:
    action = str(payload.get("action", "")).strip().upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported action: {action or '<missing>'}")

    if action == "KG_QUERY":
        relation = str(payload.get("relation", "")).strip()
        if not relation:
            raise ValueError("KG_QUERY requires a non-empty relation")
        frontier = str(payload.get("frontier", "CURRENT_FRONTIER")).strip() or "CURRENT_FRONTIER"
        if frontier != "CURRENT_FRONTIER":
            raise ValueError("Only frontier='CURRENT_FRONTIER' is supported")
        entity = payload.get("entity")
        return KGQueryAction(
            action="KG_QUERY",
            relation=relation,
            direction=normalize_direction(payload.get("direction", "auto")),
            entity=str(entity).strip() if entity not in {None, ""} else None,
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
