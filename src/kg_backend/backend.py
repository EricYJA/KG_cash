"""Backend protocol and uncached adjacency-backed implementation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from kg_backend.errors import EntityNotFoundError, RelationNotFoundError
from kg_backend.index import AdjacencyIndex, build_adjacency_index
from kg_backend.loader import load_graph_data
from kg_backend.types import (
    Direction,
    ExtractSubgraphQuery,
    FollowPathQuery,
    FollowPathResult,
    GetInRelationsQuery,
    GetNeighborsQuery,
    GetOutRelationsQuery,
    GraphDirection,
    GraphStats,
    KGQuery,
    NeighborsResult,
    PathStep,
    RelationsResult,
    SubgraphEdge,
    SubgraphResult,
)


class KGBackend(Protocol):
    """Stable typed interface for KG retrieval backends."""

    def get_out_relations(self, entity_id: str) -> list[str]:
        """Return sorted outgoing relation identifiers for one entity."""

    def get_in_relations(self, entity_id: str) -> list[str]:
        """Return sorted incoming relation identifiers for one entity."""

    def get_neighbors(
        self, entity_id: str, relation_id: str, direction: Direction = "out"
    ) -> list[str]:
        """Return sorted neighbor identifiers for one entity, relation, and direction."""

    def follow_path(
        self, start_entities: Iterable[str], path: Sequence[PathStep]
    ) -> FollowPathResult:
        """Follow a deterministic multi-step path from a set of start entities."""

    def extract_subgraph(
        self,
        seed_entities: Iterable[str],
        max_hops: int,
        allowed_relations: set[str] | None = None,
        direction: GraphDirection = "both",
        max_edges: int | None = None,
    ) -> SubgraphResult:
        """Extract a bounded subgraph around seed entities."""

    def execute(self, query: KGQuery) -> dict[str, object]:
        """Dispatch a typed query and return a dict-like payload."""

    def stats(self) -> GraphStats:
        """Return immutable backend statistics."""


class UncachedKGBackend:
    """Uncached backend backed by immutable one-hop adjacency indexes."""

    def __init__(self, index: AdjacencyIndex, data_path: str | Path) -> None:
        """Create a backend from a pre-built adjacency index."""

        self._index = index
        self._data_path = str(Path(data_path).expanduser().resolve())

    @classmethod
    def from_data_path(cls, data_path: str | Path) -> UncachedKGBackend:
        """Load graph files and build the immutable adjacency index once."""

        graph_data = load_graph_data(data_path)
        index = build_adjacency_index(graph_data)
        return cls(index=index, data_path=data_path)

    def get_out_relations(self, entity_id: str) -> list[str]:
        """Return sorted outgoing relation identifiers for one entity."""

        entity_idx = self._require_entity(entity_id)
        return [self._index.relation_ids[idx] for idx in self._index.out_relations[entity_idx]]

    def get_in_relations(self, entity_id: str) -> list[str]:
        """Return sorted incoming relation identifiers for one entity."""

        entity_idx = self._require_entity(entity_id)
        return [self._index.relation_ids[idx] for idx in self._index.in_relations[entity_idx]]

    def get_neighbors(
        self, entity_id: str, relation_id: str, direction: Direction = "out"
    ) -> list[str]:
        """Return sorted neighbor identifiers for one entity, relation, and direction."""

        entity_idx = self._require_entity(entity_id)
        relation_idx = self._require_relation(relation_id)
        if direction not in {"out", "in"}:
            raise ValueError(f"Unsupported direction: {direction}")
        adjacency = self._index.out_adj if direction == "out" else self._index.in_adj
        return [self._index.entity_ids[idx] for idx in adjacency[entity_idx].get(relation_idx, ())]

    def follow_path(
        self, start_entities: Iterable[str], path: Sequence[PathStep]
    ) -> FollowPathResult:
        """Follow a deterministic sequence of relation steps from start entities."""

        start_entity_ids = tuple(sorted(set(start_entities)))
        current_entities = {self._require_entity(entity_id) for entity_id in start_entity_ids}

        for step in path:
            relation_idx = self._require_relation(step.relation_id)
            adjacency = self._index.out_adj if step.direction == "out" else self._index.in_adj
            next_entities: set[int] = set()
            for entity_idx in sorted(current_entities):
                next_entities.update(adjacency[entity_idx].get(relation_idx, ()))
            current_entities = next_entities

        return FollowPathResult(
            start_entities=start_entity_ids,
            path=tuple(path),
            result_entities=tuple(self._index.entity_ids[idx] for idx in sorted(current_entities)),
        )

    def extract_subgraph(
        self,
        seed_entities: Iterable[str],
        max_hops: int,
        allowed_relations: set[str] | None = None,
        direction: GraphDirection = "both",
        max_edges: int | None = None,
    ) -> SubgraphResult:
        """Extract a deterministic hop-bounded subgraph from seed entities."""

        seed_entity_ids = tuple(sorted(set(seed_entities)))
        visited_entities = {self._require_entity(entity_id) for entity_id in seed_entity_ids}
        frontier = tuple(sorted(visited_entities))
        allowed_relation_indices = (
            None
            if allowed_relations is None
            else {self._require_relation(relation_id) for relation_id in allowed_relations}
        )

        seen_edges: set[tuple[int, int, int]] = set()
        truncated = False

        for _ in range(max_hops):
            if not frontier or truncated:
                break
            next_frontier: set[int] = set()
            for entity_idx in frontier:
                for triple, discovered_idx in self._iter_edges(
                    entity_idx=entity_idx,
                    direction=direction,
                    allowed_relations=allowed_relation_indices,
                ):
                    if triple in seen_edges:
                        continue
                    if max_edges is not None and len(seen_edges) >= max_edges:
                        truncated = True
                        break
                    seen_edges.add(triple)
                    if discovered_idx not in visited_entities:
                        next_frontier.add(discovered_idx)
                if truncated:
                    break
            frontier = tuple(sorted(next_frontier))
            visited_entities.update(next_frontier)

        sorted_edges = tuple(
            SubgraphEdge(
                head=self._index.entity_ids[head_idx],
                relation=self._index.relation_ids[relation_idx],
                tail=self._index.entity_ids[tail_idx],
            )
            for head_idx, relation_idx, tail_idx in sorted(seen_edges)
        )

        return SubgraphResult(
            seed_entities=seed_entity_ids,
            max_hops=max_hops,
            direction=direction,
            allowed_relations=None
            if allowed_relations is None
            else tuple(sorted(allowed_relations)),
            max_edges=max_edges,
            truncated=truncated,
            entities=tuple(self._index.entity_ids[idx] for idx in sorted(visited_entities)),
            edges=sorted_edges,
        )

    def execute(self, query: KGQuery) -> dict[str, object]:
        """Dispatch a typed query and return a dict-like payload."""

        if isinstance(query, GetOutRelationsQuery):
            return RelationsResult(
                op=query.op,
                entity_id=query.entity_id,
                relations=tuple(self.get_out_relations(query.entity_id)),
            ).model_dump(mode="python")
        if isinstance(query, GetInRelationsQuery):
            return RelationsResult(
                op=query.op,
                entity_id=query.entity_id,
                relations=tuple(self.get_in_relations(query.entity_id)),
            ).model_dump(mode="python")
        if isinstance(query, GetNeighborsQuery):
            return NeighborsResult(
                entity_id=query.entity_id,
                relation_id=query.relation_id,
                direction=query.direction,
                neighbors=tuple(
                    self.get_neighbors(
                        entity_id=query.entity_id,
                        relation_id=query.relation_id,
                        direction=query.direction,
                    )
                ),
            ).model_dump(mode="python")
        if isinstance(query, FollowPathQuery):
            return self.follow_path(query.start_entities, query.path).model_dump(mode="python")
        if isinstance(query, ExtractSubgraphQuery):
            return self.extract_subgraph(
                seed_entities=query.seed_entities,
                max_hops=query.max_hops,
                allowed_relations=None
                if query.allowed_relations is None
                else set(query.allowed_relations),
                direction=query.direction,
                max_edges=query.max_edges,
            ).model_dump(mode="python")
        message = f"Unsupported query type: {type(query)!r}"
        raise TypeError(message)

    def stats(self) -> GraphStats:
        """Return immutable backend statistics."""

        return GraphStats(
            data_path=self._data_path,
            num_entities=len(self._index.entity_ids),
            num_relations=len(self._index.relation_ids),
            num_triples=self._index.unique_triple_count,
            raw_num_triples=self._index.raw_triple_count,
            duplicate_triples_removed=(
                self._index.raw_triple_count - self._index.unique_triple_count
            ),
            entity_metadata_loaded=self._index.entity_metadata_loaded,
            relation_metadata_loaded=self._index.relation_metadata_loaded,
        )

    def _require_entity(self, entity_id: str) -> int:
        try:
            return self._index.entity_id_to_idx[entity_id]
        except KeyError as exc:
            raise EntityNotFoundError(f"Unknown entity id: {entity_id}") from exc

    def _require_relation(self, relation_id: str) -> int:
        try:
            return self._index.relation_id_to_idx[relation_id]
        except KeyError as exc:
            raise RelationNotFoundError(f"Unknown relation id: {relation_id}") from exc

    def _iter_edges(
        self,
        entity_idx: int,
        direction: GraphDirection,
        allowed_relations: set[int] | None,
    ) -> list[tuple[tuple[int, int, int], int]]:
        expansions: list[tuple[tuple[int, int, int], int]] = []
        if direction in {"out", "both"}:
            for relation_idx in sorted(self._index.out_adj[entity_idx]):
                if allowed_relations is not None and relation_idx not in allowed_relations:
                    continue
                for tail_idx in self._index.out_adj[entity_idx][relation_idx]:
                    expansions.append(((entity_idx, relation_idx, tail_idx), tail_idx))
        if direction in {"in", "both"}:
            for relation_idx in sorted(self._index.in_adj[entity_idx]):
                if allowed_relations is not None and relation_idx not in allowed_relations:
                    continue
                for head_idx in self._index.in_adj[entity_idx][relation_idx]:
                    expansions.append(((head_idx, relation_idx, entity_idx), head_idx))
        return expansions
