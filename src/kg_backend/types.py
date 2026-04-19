"""Shared query, result, and dataset schemas for the KG backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Direction: TypeAlias = Literal["out", "in"]
GraphDirection: TypeAlias = Literal["out", "in", "both"]


@dataclass(frozen=True, slots=True)
class TripleRecord:
    """A single string-identifier triple loaded from local files."""

    head: str
    relation: str
    tail: str


@dataclass(frozen=True, slots=True)
class GraphData:
    """Raw graph data loaded from local files before index construction."""

    source_path: Path
    triples_path: Path
    triples: tuple[TripleRecord, ...]
    entity_labels: dict[str, str]
    relation_labels: dict[str, str]
    entity_metadata_loaded: bool
    relation_metadata_loaded: bool


class FrozenModel(BaseModel):
    """Base class for immutable Pydantic models."""

    model_config = ConfigDict(frozen=True)


class PathStep(FrozenModel):
    """One relation traversal step used by follow-path queries."""

    relation_id: str
    direction: Direction = "out"


class GetOutRelationsQuery(FrozenModel):
    """Typed query for outgoing relations from a single entity."""

    op: Literal["get_out_relations"] = "get_out_relations"
    entity_id: str


class GetInRelationsQuery(FrozenModel):
    """Typed query for incoming relations to a single entity."""

    op: Literal["get_in_relations"] = "get_in_relations"
    entity_id: str


class GetNeighborsQuery(FrozenModel):
    """Typed query for neighbors across one relation and direction."""

    op: Literal["get_neighbors"] = "get_neighbors"
    entity_id: str
    relation_id: str
    direction: Direction = "out"


class FollowPathQuery(FrozenModel):
    """Typed query for repeated path traversal across a sequence of steps."""

    op: Literal["follow_path"] = "follow_path"
    start_entities: tuple[str, ...]
    path: tuple[PathStep, ...]


class ExtractSubgraphQuery(FrozenModel):
    """Typed query for bounded subgraph extraction around seed entities."""

    op: Literal["extract_subgraph"] = "extract_subgraph"
    seed_entities: tuple[str, ...]
    max_hops: int = Field(ge=0)
    allowed_relations: tuple[str, ...] | None = None
    direction: GraphDirection = "both"
    max_edges: int | None = Field(default=None, ge=1)


KGQuery: TypeAlias = Annotated[
    GetOutRelationsQuery
    | GetInRelationsQuery
    | GetNeighborsQuery
    | FollowPathQuery
    | ExtractSubgraphQuery,
    Field(discriminator="op"),
]

QUERY_ADAPTER = TypeAdapter(KGQuery)


def parse_query(payload: object) -> KGQuery:
    """Validate a raw payload into the supported typed query union."""

    return QUERY_ADAPTER.validate_python(payload)


class RelationsResult(FrozenModel):
    """Result payload for one-hop relation listing operations."""

    op: Literal["get_out_relations", "get_in_relations"]
    entity_id: str
    relations: tuple[str, ...]


class NeighborsResult(FrozenModel):
    """Result payload for one-hop neighbor expansion."""

    op: Literal["get_neighbors"] = "get_neighbors"
    entity_id: str
    relation_id: str
    direction: Direction
    neighbors: tuple[str, ...]


class FollowPathResult(FrozenModel):
    """Result payload for path following."""

    op: Literal["follow_path"] = "follow_path"
    start_entities: tuple[str, ...]
    path: tuple[PathStep, ...]
    result_entities: tuple[str, ...]


class SubgraphEdge(FrozenModel):
    """One directed edge in an extracted subgraph."""

    head: str
    relation: str
    tail: str


class SubgraphResult(FrozenModel):
    """Result payload for bounded subgraph extraction."""

    op: Literal["extract_subgraph"] = "extract_subgraph"
    seed_entities: tuple[str, ...]
    max_hops: int
    direction: GraphDirection
    allowed_relations: tuple[str, ...] | None
    max_edges: int | None
    truncated: bool
    entities: tuple[str, ...]
    edges: tuple[SubgraphEdge, ...]


class GraphStats(FrozenModel):
    """Basic graph statistics exposed by the backend and API."""

    data_path: str
    num_entities: int
    num_relations: int
    num_triples: int
    raw_num_triples: int
    duplicate_triples_removed: int
    entity_metadata_loaded: bool
    relation_metadata_loaded: bool
