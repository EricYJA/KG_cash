from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from kg_backend.backend import UncachedKGBackend
from kg_backend.errors import EntityNotFoundError, RelationNotFoundError
from kg_backend.types import GraphStats

from .config import LLMFrontendConfig
from .schemas import (
    BackendQueryResult,
    FrontierObservation,
    KGQueryAction,
    unique_strings,
)


class KGBackendAdapter:
    """Thin llm_frontend adapter over the uncached KG backend."""

    def __init__(self, backend: UncachedKGBackend, config: LLMFrontendConfig) -> None:
        self.backend = backend
        self.config = config

    @classmethod
    def from_path(
        cls,
        data_path: str | Path,
        config: LLMFrontendConfig,
    ) -> KGBackendAdapter:
        return cls(backend=UncachedKGBackend.from_data_path(data_path), config=config)

    def stats(self) -> GraphStats:
        return self.backend.stats()

    def resolve_initial_frontier(self, seed_text: str | None) -> list[str]:
        if not seed_text:
            return []
        entity_id = str(seed_text).strip()
        if not entity_id:
            return []
        try:
            self.backend.get_out_relations(entity_id)
        except EntityNotFoundError:
            pass
        else:
            return [entity_id]
        if not self.backend.entity_name_exists(entity_id):
            return []
        return unique_strings(
            self.backend.search_entity_ids_by_name(
                entity_id,
                limit=self.config.max_frontier_entities,
            )
        )

    def describe_frontier(self, frontier: list[str]) -> FrontierObservation:
        unique_frontier = unique_strings(frontier)
        sample_entities = unique_frontier[: self.config.max_frontier_entities]
        scan_entities = unique_frontier[: self.config.relation_scan_limit]

        forward_counts: Counter[str] = Counter()
        backward_counts: Counter[str] = Counter()
        for entity_id in scan_entities:
            try:
                for relation_id in self.backend.get_out_relations(entity_id):
                    forward_counts[relation_id] += 1
                for relation_id in self.backend.get_in_relations(entity_id):
                    backward_counts[relation_id] += 1
            except EntityNotFoundError:
                continue

        return FrontierObservation(
            frontier=unique_frontier,
            frontier_size=len(unique_frontier),
            sample_entities=sample_entities,
            forward_relations=[
                relation_id
                for relation_id, _ in forward_counts.most_common(
                    self.config.max_relation_candidates
                )
            ],
            backward_relations=[
                relation_id
                for relation_id, _ in backward_counts.most_common(
                    self.config.max_relation_candidates
                )
            ],
        )

    def execute_query(
        self,
        current_frontier: list[str],
        action: KGQueryAction,
    ) -> BackendQueryResult:
        input_frontier = unique_strings(current_frontier)

        frontier_after_hop: list[str] = []
        resolved_direction = "empty"
        primitive_calls = 0

        if input_frontier:
            if action.direction == "auto":
                frontier_after_hop, resolved_direction, primitive_calls = (
                    self._execute_auto_hop(
                        input_frontier,
                        action.relation,
                    )
                )
            else:
                hop_direction: Literal["out", "in"] = (
                    "out" if action.direction == "forward" else "in"
                )
                frontier_after_hop, primitive_calls = self._collect_neighbors(
                    input_frontier,
                    action.relation,
                    hop_direction,
                )
                if frontier_after_hop:
                    resolved_direction = action.direction

        output_frontier = unique_strings(frontier_after_hop)
        observation = self.describe_frontier(output_frontier)
        return BackendQueryResult(
            resolved_direction=resolved_direction,
            output_frontier=output_frontier,
            observation=observation,
        )

    def _execute_auto_hop(
        self,
        frontier: list[str],
        relation_id: str,
    ) -> tuple[list[str], str, int]:
        forward_frontier, forward_calls = self._collect_neighbors(
            frontier, relation_id, "out"
        )
        if forward_frontier:
            return forward_frontier, "forward", forward_calls
        backward_frontier, backward_calls = self._collect_neighbors(
            frontier, relation_id, "in"
        )
        if backward_frontier:
            return backward_frontier, "backward", forward_calls + backward_calls
        return [], "empty", forward_calls + backward_calls

    def _collect_neighbors(
        self,
        frontier: list[str],
        relation_id: str,
        direction: Literal["out", "in"],
    ) -> tuple[list[str], int]:
        neighbors: set[str] = set()
        primitive_calls = 0
        for entity_id in unique_strings(frontier):
            try:
                entity_neighbors = self.backend.get_neighbors(
                    entity_id, relation_id, direction=direction
                )
            except EntityNotFoundError:
                continue
            except RelationNotFoundError:
                return [], primitive_calls
            primitive_calls += 1
            neighbors.update(entity_neighbors)
        return sorted(neighbors), primitive_calls
