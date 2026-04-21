"""Backend protocol and uncached/cached adjacency-backed implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from kg_backend.errors import EntityNotFoundError, RelationNotFoundError
from kg_backend.index import AdjacencyIndex, build_adjacency_index
from kg_backend.loader import load_graph_data
from kg_backend.name_lookup import build_entity_name_index
from kg_backend.types import (
    Direction,
    EntityNameExistsQuery,
    EntityNameExistsResult,
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
    SearchEntityIdsByNameQuery,
    SearchEntityIdsByNameResult,
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

    def search_entity_ids_by_name(self, name: str, limit: int = 10) -> list[str]:
        """Return deterministic entity ids for one exact normalized entity name."""

    def entity_name_exists(self, name: str) -> bool:
        """Return whether one exact normalized entity name exists in the backend."""

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
        self._entity_name_index = build_entity_name_index(
            self._data_path,
            self._index.entity_labels,
        )

    @classmethod
    def from_data_path(cls, data_path: str | Path) -> UncachedKGBackend:
        """Load graph files and build the immutable adjacency index once."""

        graph_data = load_graph_data(data_path)
        index = build_adjacency_index(graph_data)
        return cls(index=index, data_path=data_path)

    def get_out_relations(self, entity_id: str) -> list[str]:
        """Return sorted outgoing relation identifiers for one entity."""

        entity_idx = self._require_entity(entity_id)
        return [
            self._index.relation_ids[idx]
            for idx in self._index.out_relations[entity_idx]
        ]

    def get_in_relations(self, entity_id: str) -> list[str]:
        """Return sorted incoming relation identifiers for one entity."""

        entity_idx = self._require_entity(entity_id)
        return [
            self._index.relation_ids[idx]
            for idx in self._index.in_relations[entity_idx]
        ]

    def get_neighbors(
        self, entity_id: str, relation_id: str, direction: Direction = "out"
    ) -> list[str]:
        """Return sorted neighbor identifiers for one entity, relation, and direction."""

        entity_idx = self._require_entity(entity_id)
        relation_idx = self._require_relation(relation_id)
        if direction not in {"out", "in"}:
            raise ValueError(f"Unsupported direction: {direction}")
        adjacency = self._index.out_adj if direction == "out" else self._index.in_adj
        return [
            self._index.entity_ids[idx]
            for idx in adjacency[entity_idx].get(relation_idx, ())
        ]

    def search_entity_ids_by_name(self, name: str, limit: int = 10) -> list[str]:
        """Return deterministic entity ids for one exact normalized entity name."""

        return self._entity_name_index.search_entity_ids_by_name(name, limit=limit)

    def entity_name_exists(self, name: str) -> bool:
        """Return whether one exact normalized entity name exists in the backend."""

        return self._entity_name_index.entity_name_exists(name)

    def follow_path(
        self, start_entities: Iterable[str], path: Sequence[PathStep]
    ) -> FollowPathResult:
        """Follow a deterministic sequence of relation steps from start entities."""

        start_entity_ids = tuple(sorted(set(start_entities)))
        current_entities = {
            self._require_entity(entity_id) for entity_id in start_entity_ids
        }

        for step in path:
            relation_idx = self._require_relation(step.relation_id)
            adjacency = (
                self._index.out_adj if step.direction == "out" else self._index.in_adj
            )
            next_entities: set[int] = set()
            for entity_idx in sorted(current_entities):
                next_entities.update(adjacency[entity_idx].get(relation_idx, ()))
            current_entities = next_entities

        return FollowPathResult(
            start_entities=start_entity_ids,
            path=tuple(path),
            result_entities=tuple(
                self._index.entity_ids[idx] for idx in sorted(current_entities)
            ),
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
        visited_entities = {
            self._require_entity(entity_id) for entity_id in seed_entity_ids
        }
        frontier = tuple(sorted(visited_entities))
        allowed_relation_indices = (
            None
            if allowed_relations is None
            else {
                self._require_relation(relation_id) for relation_id in allowed_relations
            }
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
            entities=tuple(
                self._index.entity_ids[idx] for idx in sorted(visited_entities)
            ),
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
            return self.follow_path(query.start_entities, query.path).model_dump(
                mode="python"
            )
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
        if isinstance(query, SearchEntityIdsByNameQuery):
            return SearchEntityIdsByNameResult(
                name=query.name,
                entity_ids=tuple(
                    self.search_entity_ids_by_name(
                        name=query.name,
                        limit=query.limit,
                    )
                ),
            ).model_dump(mode="python")
        if isinstance(query, EntityNameExistsQuery):
            return EntityNameExistsResult(
                name=query.name,
                exists=self.entity_name_exists(query.name),
            ).model_dump(mode="python")
        message = f"Unsupported query type: {type(query)!r}"
        raise TypeError(message)

    def get_neighborhood(
        self,
        entity_ids: list[str],
        *,
        excluded_relations: frozenset[str] = frozenset(),
        scan_limit: int = 12,
        max_relations: int = 12,
        max_neighbors_per_relation: int = 5,
    ) -> list[tuple[str, list[str]]]:
        """Return top outgoing (relation, [neighbors]) pairs for a set of entities."""
        scan_entities = entity_ids[:scan_limit]

        relation_counts: Counter[int] = Counter()
        for entity_id in scan_entities:
            try:
                entity_idx = self._index.entity_id_to_idx[entity_id]
            except KeyError:
                continue
            for relation_idx in self._index.out_adj[entity_idx]:
                relation_id = self._index.relation_ids[relation_idx]
                if relation_id not in excluded_relations:
                    relation_counts[relation_idx] += 1

        result: list[tuple[str, list[str]]] = []
        for relation_idx, _ in relation_counts.most_common(max_relations):
            neighbors: set[str] = set()
            for entity_id in scan_entities:
                try:
                    entity_idx = self._index.entity_id_to_idx[entity_id]
                except KeyError:
                    continue
                for tail_idx in self._index.out_adj[entity_idx].get(relation_idx, ()):
                    neighbors.add(self._index.entity_ids[tail_idx])
            capped = sorted(neighbors)[:max_neighbors_per_relation]
            if capped:
                result.append((self._index.relation_ids[relation_idx], capped))
        return result

    def get_entity_label(self, entity_id: str) -> str | None:
        return self._index.entity_labels.get(entity_id)

    def is_cvt_node(self, entity_id: str) -> bool:
        if not (entity_id.startswith("m.") or entity_id.startswith("g.")):
            return False
        label = self._index.entity_labels.get(entity_id)
        return label is None or label == entity_id

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
                if (
                    allowed_relations is not None
                    and relation_idx not in allowed_relations
                ):
                    continue
                for tail_idx in self._index.out_adj[entity_idx][relation_idx]:
                    expansions.append(((entity_idx, relation_idx, tail_idx), tail_idx))
        if direction in {"in", "both"}:
            for relation_idx in sorted(self._index.in_adj[entity_idx]):
                if (
                    allowed_relations is not None
                    and relation_idx not in allowed_relations
                ):
                    continue
                for head_idx in self._index.in_adj[entity_idx][relation_idx]:
                    expansions.append(((head_idx, relation_idx, entity_idx), head_idx))
        return expansions


_EntityAdj = dict[str, tuple[str, ...]]


class CachePolicy(ABC):
    """Abstract base for per-entity neighborhood cache replacement policies."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._requests: int = 0
        self._hits: int = 0

    def get(self, key: str) -> _EntityAdj | None:
        """Return cached adjacency for key (or None on miss), tracking hit/miss stats."""
        result = self._lookup(key)
        self._requests += 1
        if result is not None:
            self._hits += 1
        return result

    @abstractmethod
    def _lookup(self, key: str) -> _EntityAdj | None:
        """Policy-specific cache lookup; return value or None."""

    @abstractmethod
    def put(self, key: str, value: _EntityAdj) -> None:
        """Insert key→value, evicting one entry if the cache is full."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    def info(self) -> dict[str, object]:
        hit_rate = self._hits / self._requests if self._requests > 0 else 0.0
        return {
            "size": len(self),
            "maxsize": self._maxsize,
            "requests": self._requests,
            "hits": self._hits,
            "misses": self._requests - self._hits,
            "hit_rate": round(hit_rate, 4),
        }


class LRUPolicy(CachePolicy):
    """Least-recently-used eviction backed by OrderedDict."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self._cache: OrderedDict[str, _EntityAdj] = OrderedDict()

    def _lookup(self, key: str) -> _EntityAdj | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: _EntityAdj) -> None:
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


class LFUPolicy(CachePolicy):
    """Least-frequently-used eviction. Ties broken by insertion order."""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self._cache: dict[str, _EntityAdj] = {}
        self._freq: Counter[str] = Counter()

    def _lookup(self, key: str) -> _EntityAdj | None:
        if key not in self._cache:
            return None
        self._freq[key] += 1
        return self._cache[key]

    def put(self, key: str, value: _EntityAdj) -> None:
        if len(self._cache) >= self._maxsize:
            lfu_key = min(self._freq, key=lambda k: self._freq[k])
            del self._cache[lfu_key]
            del self._freq[lfu_key]
        self._cache[key] = value
        self._freq[key] = 1

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._freq.clear()


class OraclePolicy(CachePolicy):
    """Pre-populated, frozen cache. Top-K entities are loaded upfront; no changes at runtime."""

    def __init__(self, preloaded: dict[str, _EntityAdj]) -> None:
        super().__init__(maxsize=len(preloaded))
        self._cache: dict[str, _EntityAdj] = preloaded

    @classmethod
    def from_index(
        cls,
        index: AdjacencyIndex,
        entity_ids: list[str],
        maxsize: int,
    ) -> OraclePolicy:
        """Build adjacency for up to maxsize entities from the KG index."""
        preloaded: dict[str, _EntityAdj] = {}
        for entity_id in entity_ids:
            if len(preloaded) >= maxsize:
                break
            try:
                entity_idx = index.entity_id_to_idx[entity_id]
            except KeyError:
                continue
            preloaded[entity_id] = {
                index.relation_ids[relation_idx]: tuple(
                    index.entity_ids[tail_idx] for tail_idx in tail_indices
                )
                for relation_idx, tail_indices in index.out_adj[entity_idx].items()
            }
        return cls(preloaded)

    def _lookup(self, key: str) -> _EntityAdj | None:
        return self._cache.get(key)

    def put(self, key: str, value: _EntityAdj) -> None:
        pass  # frozen: no new entries at runtime

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


class CachedKGBackend(UncachedKGBackend):
    """KG backend with a pluggable per-entity cache over outgoing adjacency."""

    def __init__(
        self,
        index: AdjacencyIndex,
        data_path: str | Path,
        policy: CachePolicy,
    ) -> None:
        super().__init__(index, data_path)
        self._policy = policy

    @classmethod
    def from_data_path(  # type: ignore[override]
        cls,
        data_path: str | Path,
        policy: CachePolicy,
    ) -> CachedKGBackend:
        graph_data = load_graph_data(data_path)
        index = build_adjacency_index(graph_data)
        return cls(index=index, data_path=data_path, policy=policy)

    @classmethod
    def with_oracle_policy(
        cls,
        data_path: str | Path,
        top_entity_ids: list[str],
        cache_size: int,
    ) -> CachedKGBackend:
        """Load the KG once and pre-populate an Oracle policy with top-frequency entities."""
        graph_data = load_graph_data(data_path)
        index = build_adjacency_index(graph_data)
        policy = OraclePolicy.from_index(index, top_entity_ids, cache_size)
        return cls(index=index, data_path=data_path, policy=policy)

    def _get_entity_adjacency(self, entity_id: str) -> _EntityAdj:
        """Return outgoing adjacency for one entity, consulting the cache first."""
        cached = self._policy.get(entity_id)
        if cached is not None:
            return cached

        try:
            entity_idx = self._index.entity_id_to_idx[entity_id]
        except KeyError:
            return {}

        adjacency: _EntityAdj = {
            self._index.relation_ids[relation_idx]: tuple(
                self._index.entity_ids[tail_idx]
                for tail_idx in tail_indices
            )
            for relation_idx, tail_indices in self._index.out_adj[entity_idx].items()
        }
        self._policy.put(entity_id, adjacency)
        return adjacency

    def get_neighborhood(
        self,
        entity_ids: list[str],
        *,
        excluded_relations: frozenset[str] = frozenset(),
        scan_limit: int = 12,
        max_relations: int = 12,
        max_neighbors_per_relation: int = 5,
    ) -> list[tuple[str, list[str]]]:
        scan_entities = entity_ids[:scan_limit]

        entity_adjacencies = {
            entity_id: self._get_entity_adjacency(entity_id)
            for entity_id in scan_entities
        }

        relation_counts: Counter[str] = Counter()
        for adjacency in entity_adjacencies.values():
            for relation_id in adjacency:
                if relation_id not in excluded_relations:
                    relation_counts[relation_id] += 1

        result: list[tuple[str, list[str]]] = []
        for relation_id, _ in relation_counts.most_common(max_relations):
            neighbors: set[str] = set()
            for adjacency in entity_adjacencies.values():
                neighbors.update(adjacency.get(relation_id, ()))
            capped = sorted(neighbors)[:max_neighbors_per_relation]
            if capped:
                result.append((relation_id, capped))
        return result

    def cache_info(self) -> dict[str, object]:
        return self._policy.info()

    def cache_clear(self) -> None:
        self._policy.clear()
