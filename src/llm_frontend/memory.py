from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import BackendQueryResult, KGQueryAction, unique_strings


def _preview(values: list[str], limit: int = 4) -> str:
    if not values:
        return "[]"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else ", ..."
    return "[" + ", ".join(shown) + suffix + "]"


@dataclass(frozen=True)
class MemoryTurn:
    step_id: int
    relation: str
    requested_direction: str
    resolved_direction: str
    frontier_size: int
    sample_entities: list[str] = field(default_factory=list)
    reason: str | None = None

    def as_line(self) -> str:
        parts = [
            f"{self.step_id}. relation={self.relation}",
            f"requested={self.requested_direction}",
            f"actual={self.resolved_direction}",
            f"frontier_size={self.frontier_size}",
            f"sample={_preview(self.sample_entities)}",
        ]
        if self.reason:
            parts.append(f"reason={self.reason}")
        return " | ".join(parts)


@dataclass
class PlannerMemory:
    max_steps: int
    max_history: int
    current_frontier: list[str] = field(default_factory=list)
    turns: list[MemoryTurn] = field(default_factory=list)
    failed_initial_entities: list[str] = field(default_factory=list)

    def set_frontier(self, frontier: list[str]) -> None:
        self.current_frontier = unique_strings(frontier)

    def frontier_signature(self, frontier: list[str] | None = None) -> str:
        values = unique_strings(
            frontier if frontier is not None else self.current_frontier
        )
        return "|".join(values) if values else "<empty>"

    def record_query(
        self,
        step_id: int,
        action: KGQueryAction,
        result: BackendQueryResult,
    ) -> None:
        self.set_frontier(result.output_frontier)
        self.turns.append(
            MemoryTurn(
                step_id=step_id,
                relation=action.relation,
                requested_direction=action.direction,
                resolved_direction=result.resolved_direction,
                frontier_size=result.observation.frontier_size,
                sample_entities=result.observation.sample_entities,
                reason=action.reason,
            )
        )
        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history :]

    @property
    def steps_used(self) -> int:
        return len(self.turns)

    @property
    def steps_remaining(self) -> int:
        return max(self.max_steps - self.steps_used, 0)

    def record_failed_initial_entity(self, entity: str) -> None:
        text = str(entity).strip()
        if not text:
            return
        self.failed_initial_entities.append(text)
        if len(self.failed_initial_entities) > self.max_history:
            self.failed_initial_entities = self.failed_initial_entities[
                -self.max_history :
            ]

    def format_failed_initial_entities(self) -> str:
        if not self.failed_initial_entities:
            return "None."
        return ", ".join(self.failed_initial_entities)

    def format_history(self) -> str:
        if not self.turns:
            return "No prior KG queries."
        return "\n".join(turn.as_line() for turn in self.turns)
