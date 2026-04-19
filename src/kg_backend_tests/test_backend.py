"""Tests for the uncached adjacency backend."""

from __future__ import annotations

import pytest

from kg_backend.backend import UncachedKGBackend
from kg_backend.errors import EntityNotFoundError, RelationNotFoundError
from kg_backend.types import GetNeighborsQuery, PathStep


def test_duplicate_triples_are_removed_in_indexes(backend: UncachedKGBackend) -> None:
    stats = backend.stats()

    assert stats.raw_num_triples == 8
    assert stats.num_triples == 7
    assert stats.duplicate_triples_removed == 1


def test_deterministic_relations_and_neighbors(backend: UncachedKGBackend) -> None:
    assert backend.get_out_relations("alice") == ["knows", "likes"]
    assert backend.get_in_relations("bob") == ["likes", "parent"]
    assert backend.get_neighbors("alice", "likes", direction="out") == ["bob", "carol"]


def test_empty_results_for_known_ids(backend: UncachedKGBackend) -> None:
    assert backend.get_neighbors("erin", "knows", direction="out") == []


def test_missing_query_ids_raise(backend: UncachedKGBackend) -> None:
    with pytest.raises(EntityNotFoundError):
        backend.get_out_relations("missing")

    with pytest.raises(RelationNotFoundError):
        backend.get_neighbors("alice", "missing", direction="out")


def test_follow_path_with_mixed_directions(backend: UncachedKGBackend) -> None:
    result = backend.follow_path(
        ["alice"],
        [
            PathStep(relation_id="likes", direction="out"),
            PathStep(relation_id="parent", direction="in"),
        ],
    )

    assert result.result_entities == ("carol",)


def test_extract_subgraph_correctness(backend: UncachedKGBackend) -> None:
    result = backend.extract_subgraph(["alice"], max_hops=1, direction="both")

    assert result.entities == ("alice", "bob", "carol", "dave", "erin")
    assert [(edge.head, edge.relation, edge.tail) for edge in result.edges] == [
        ("alice", "knows", "erin"),
        ("alice", "likes", "bob"),
        ("alice", "likes", "carol"),
        ("dave", "knows", "alice"),
    ]


def test_execute_dispatches_typed_query(backend: UncachedKGBackend) -> None:
    result = backend.execute(
        GetNeighborsQuery(entity_id="alice", relation_id="likes", direction="out")
    )

    assert result == {
        "op": "get_neighbors",
        "entity_id": "alice",
        "relation_id": "likes",
        "direction": "out",
        "neighbors": ("bob", "carol"),
    }
