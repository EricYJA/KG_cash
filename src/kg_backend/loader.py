"""Load triples and optional metadata from local TSV or Parquet files."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from kg_backend.errors import KGDataError, MetadataValidationError
from kg_backend.types import GraphData, TripleRecord

TRIPLE_COLUMNS = ("head", "relation", "tail")
METADATA_COLUMNS = ("id", "label")


def load_graph_data(source: str | Path) -> GraphData:
    """Load triples and optional metadata from a directory or triples file."""

    source_path = Path(source).expanduser().resolve()
    triples_path, entities_path, relations_path = _resolve_graph_paths(source_path)
    triples = _load_triples(triples_path)
    entity_labels = _load_optional_metadata(entities_path, "entities")
    relation_labels = _load_optional_metadata(relations_path, "relations")
    _validate_metadata_ids(triples, entity_labels, relation_labels)
    return GraphData(
        source_path=source_path,
        triples_path=triples_path,
        triples=triples,
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        entity_metadata_loaded=entities_path is not None,
        relation_metadata_loaded=relations_path is not None,
    )


def _resolve_graph_paths(path: Path) -> tuple[Path, Path | None, Path | None]:
    if path.is_dir():
        base_dir = path
        triples_path = _pick_one(base_dir, "triples", required=True)
        if triples_path is None:
            raise KGDataError(
                f"Expected either triples.tsv or triples.parquet in {base_dir}."
            )
    elif path.is_file():
        if path.suffix not in {".tsv", ".parquet"}:
            raise KGDataError(
                "Triples input must be a directory or a TSV or Parquet triples file."
            )
        base_dir = path.parent
        triples_path = path
    else:
        raise KGDataError(f"Graph input path does not exist: {path}")
    entities_path = _pick_one(base_dir, "entities", required=False)
    relations_path = _pick_one(base_dir, "relations", required=False)
    return triples_path, entities_path, relations_path


def _pick_one(directory: Path, stem: str, required: bool) -> Path | None:
    candidates = [directory / f"{stem}.tsv", directory / f"{stem}.parquet"]
    present = [candidate for candidate in candidates if candidate.exists()]
    if len(present) > 1:
        raise KGDataError(
            f"Expected at most one {stem} file in {directory}, found: "
            f"{', '.join(path.name for path in present)}."
        )
    if required and not present:
        raise KGDataError(
            f"Expected either {stem}.tsv or {stem}.parquet in {directory}."
        )
    return present[0] if present else None


def _load_triples(path: Path) -> tuple[TripleRecord, ...]:
    rows = _load_rows(path, TRIPLE_COLUMNS, kind="triples")
    triples: list[TripleRecord] = []
    for row in rows:
        triples.append(
            TripleRecord(
                head=_require_value(row, "head", path),
                relation=_require_value(row, "relation", path),
                tail=_require_value(row, "tail", path),
            )
        )
    if not triples:
        raise KGDataError(f"No triples were found in {path}.")
    return tuple(triples)


def _load_optional_metadata(path: Path | None, kind: str) -> dict[str, str]:
    if path is None:
        return {}
    rows = _load_rows(path, METADATA_COLUMNS, kind=kind)
    labels: dict[str, str] = {}
    for row in rows:
        identifier = _require_value(row, "id", path)
        label = _require_value(row, "label", path)
        if identifier in labels:
            raise MetadataValidationError(
                f"Duplicate {kind} identifier '{identifier}' found in {path}."
            )
        labels[identifier] = label
    return labels


def _load_rows(
    path: Path, required_columns: tuple[str, ...], kind: str
) -> tuple[dict[str, str], ...]:
    if path.suffix == ".tsv":
        return _load_tsv_rows(path, required_columns, kind)
    if path.suffix == ".parquet":
        return _load_parquet_rows(path, required_columns, kind)
    raise KGDataError(f"Unsupported file type for {path}.")


def _load_tsv_rows(
    path: Path, required_columns: tuple[str, ...], kind: str
) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        _validate_columns(path, fieldnames, required_columns, kind)
        return tuple(
            {column: row[column] for column in required_columns} for row in reader
        )


def _load_parquet_rows(
    path: Path, required_columns: tuple[str, ...], kind: str
) -> tuple[dict[str, str], ...]:
    table = pq.read_table(path)
    _validate_columns(path, table.column_names, required_columns, kind)
    rows: list[dict[str, str]] = []
    for record in table.to_pylist():
        converted = {
            column: "" if record[column] is None else str(record[column])
            for column in required_columns
        }
        rows.append(converted)
    return tuple(rows)


def _validate_columns(
    path: Path,
    actual_columns: Iterable[str],
    required_columns: tuple[str, ...],
    kind: str,
) -> None:
    actual = set(actual_columns)
    missing = [column for column in required_columns if column not in actual]
    if missing:
        raise KGDataError(
            f"{kind.capitalize()} file {path} is missing required columns: {', '.join(missing)}."
        )


def _require_value(row: dict[str, str], key: str, path: Path) -> str:
    value = row[key].strip()
    if not value:
        raise KGDataError(f"Column '{key}' contains an empty value in {path}.")
    return value


def _validate_metadata_ids(
    triples: tuple[TripleRecord, ...],
    entity_labels: dict[str, str],
    relation_labels: dict[str, str],
) -> None:
    if entity_labels:
        triple_entities = {triple.head for triple in triples} | {
            triple.tail for triple in triples
        }
        missing_entities = sorted(triple_entities - entity_labels.keys())
        if missing_entities:
            raise MetadataValidationError(
                "Entity metadata is missing identifiers referenced by triples: "
                f"{', '.join(missing_entities)}."
            )
    if relation_labels:
        triple_relations = {triple.relation for triple in triples}
        missing_relations = sorted(triple_relations - relation_labels.keys())
        if missing_relations:
            raise MetadataValidationError(
                "Relation metadata is missing identifiers referenced by triples: "
                f"{', '.join(missing_relations)}."
            )
