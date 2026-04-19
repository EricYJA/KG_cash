"""Build immutable adjacency indexes from loaded graph data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from kg_backend.types import GraphData


@dataclass(frozen=True, slots=True)
class AdjacencyIndex:
    """Immutable adjacency structures and identifier maps for one graph."""

    entity_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    entity_id_to_idx: Mapping[str, int]
    relation_id_to_idx: Mapping[str, int]
    out_adj: tuple[Mapping[int, tuple[int, ...]], ...]
    in_adj: tuple[Mapping[int, tuple[int, ...]], ...]
    out_relations: tuple[tuple[int, ...], ...]
    in_relations: tuple[tuple[int, ...], ...]
    triples: tuple[tuple[int, int, int], ...]
    entity_labels: Mapping[str, str]
    relation_labels: Mapping[str, str]
    raw_triple_count: int
    unique_triple_count: int
    entity_metadata_loaded: bool
    relation_metadata_loaded: bool


def build_adjacency_index(graph_data: GraphData) -> AdjacencyIndex:
    """Build deterministic adjacency indexes from loaded graph data."""

    entity_ids = tuple(
        sorted(
            {triple.head for triple in graph_data.triples}
            | {triple.tail for triple in graph_data.triples}
            | set(graph_data.entity_labels)
        )
    )
    relation_ids = tuple(
        sorted({triple.relation for triple in graph_data.triples} | set(graph_data.relation_labels))
    )
    entity_id_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    relation_id_to_idx = {relation_id: idx for idx, relation_id in enumerate(relation_ids)}

    unique_triples = {
        (
            entity_id_to_idx[triple.head],
            relation_id_to_idx[triple.relation],
            entity_id_to_idx[triple.tail],
        )
        for triple in graph_data.triples
    }
    sorted_triples = tuple(sorted(unique_triples))

    out_builders: list[defaultdict[int, set[int]]] = [
        defaultdict(set) for _ in range(len(entity_ids))
    ]
    in_builders: list[defaultdict[int, set[int]]] = [
        defaultdict(set) for _ in range(len(entity_ids))
    ]

    for head_idx, relation_idx, tail_idx in sorted_triples:
        out_builders[head_idx][relation_idx].add(tail_idx)
        in_builders[tail_idx][relation_idx].add(head_idx)

    out_adj = tuple(_freeze_relation_map(builder) for builder in out_builders)
    in_adj = tuple(_freeze_relation_map(builder) for builder in in_builders)
    out_relations = tuple(tuple(sorted(builder)) for builder in out_builders)
    in_relations = tuple(tuple(sorted(builder)) for builder in in_builders)

    return AdjacencyIndex(
        entity_ids=entity_ids,
        relation_ids=relation_ids,
        entity_id_to_idx=MappingProxyType(entity_id_to_idx),
        relation_id_to_idx=MappingProxyType(relation_id_to_idx),
        out_adj=out_adj,
        in_adj=in_adj,
        out_relations=out_relations,
        in_relations=in_relations,
        triples=sorted_triples,
        entity_labels=MappingProxyType(dict(graph_data.entity_labels)),
        relation_labels=MappingProxyType(dict(graph_data.relation_labels)),
        raw_triple_count=len(graph_data.triples),
        unique_triple_count=len(sorted_triples),
        entity_metadata_loaded=graph_data.entity_metadata_loaded,
        relation_metadata_loaded=graph_data.relation_metadata_loaded,
    )


def _freeze_relation_map(builder: defaultdict[int, set[int]]) -> Mapping[int, tuple[int, ...]]:
    return MappingProxyType(
        {relation_idx: tuple(sorted(neighbors)) for relation_idx, neighbors in builder.items()}
    )
