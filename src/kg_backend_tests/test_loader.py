"""Tests for TSV and Parquet graph loading."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kg_backend.errors import KGDataError, MetadataValidationError
from kg_backend.loader import load_graph_data


def test_load_tsv_graph(tiny_kg_path: Path) -> None:
    graph_data = load_graph_data(tiny_kg_path)

    assert graph_data.triples_path == tiny_kg_path
    assert len(graph_data.triples) == 8
    assert graph_data.triples[0].head == "alice"
    assert graph_data.entity_metadata_loaded is False
    assert graph_data.relation_metadata_loaded is False


def test_load_parquet_graph(tmp_path: Path) -> None:
    triples_path = tmp_path / "triples.parquet"
    table = pa.table(
        {
            "head": ["alice", "bob", "carol"],
            "relation": ["likes", "likes", "parent"],
            "tail": ["bob", "dave", "bob"],
        }
    )
    pq.write_table(table, triples_path)

    graph_data = load_graph_data(tmp_path)

    assert graph_data.triples_path == triples_path
    assert [triple.relation for triple in graph_data.triples] == ["likes", "likes", "parent"]


def test_missing_entity_ids_in_metadata_raise(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "triples.tsv",
        ("head", "relation", "tail"),
        [
            ("alice", "likes", "bob"),
            ("bob", "likes", "carol"),
        ],
    )
    _write_tsv(
        tmp_path / "entities.tsv",
        ("id", "label"),
        [
            ("alice", "Alice"),
            ("carol", "Carol"),
        ],
    )

    with pytest.raises(MetadataValidationError, match="bob"):
        load_graph_data(tmp_path)


def test_missing_relation_ids_in_metadata_raise(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "triples.tsv",
        ("head", "relation", "tail"),
        [("alice", "likes", "bob")],
    )
    _write_tsv(
        tmp_path / "relations.tsv",
        ("id", "label"),
        [("parent", "parent of")],
    )

    with pytest.raises(MetadataValidationError, match="likes"):
        load_graph_data(tmp_path)


def test_invalid_triples_columns_raise(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "triples.tsv",
        ("source", "relation", "target"),
        [("alice", "likes", "bob")],
    )

    with pytest.raises(KGDataError, match="missing required columns: head, tail"):
        load_graph_data(tmp_path)


def _write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
