"""Exact entity-name lookup helpers for backend metadata."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from kg_backend.errors import KGDataError

NAME_MAPPING_COLUMNS = ("name", "id")


def normalize_entity_name(value: str) -> str:
    """Normalize one entity name for deterministic exact lookup."""

    return " ".join(str(value).split()).casefold().strip()


@dataclass(frozen=True, slots=True)
class EntityNameIndex:
    """Immutable exact entity-name lookup index."""

    normalized_to_entity_ids: Mapping[str, tuple[str, ...]]

    def search_entity_ids_by_name(self, name: str, limit: int = 10) -> list[str]:
        """Return deterministic entity ids for one exact normalized name."""

        normalized_name = normalize_entity_name(name)
        if not normalized_name or limit <= 0:
            return []
        return list(self.normalized_to_entity_ids.get(normalized_name, ())[:limit])

    def entity_name_exists(self, name: str) -> bool:
        """Return whether an exact normalized entity name exists."""

        normalized_name = normalize_entity_name(name)
        return (
            bool(normalized_name) and normalized_name in self.normalized_to_entity_ids
        )


def build_entity_name_index(
    data_path: str | Path,
    entity_labels: Mapping[str, str],
) -> EntityNameIndex:
    """Build the exact entity-name index from dataset files and entity labels."""

    source_path = Path(data_path).expanduser().resolve()
    base_dir = source_path if source_path.is_dir() else source_path.parent

    names_to_entity_ids: defaultdict[str, set[str]] = defaultdict(set)
    for name, entity_id in _load_name_rows(base_dir):
        names_to_entity_ids[normalize_entity_name(name)].add(entity_id)
    for entity_id, label in entity_labels.items():
        normalized_name = normalize_entity_name(label)
        if normalized_name:
            names_to_entity_ids[normalized_name].add(entity_id)

    return EntityNameIndex(
        normalized_to_entity_ids=MappingProxyType(
            {
                normalized_name: tuple(sorted(entity_ids))
                for normalized_name, entity_ids in sorted(names_to_entity_ids.items())
                if normalized_name
            }
        )
    )


def _load_name_rows(base_dir: Path) -> tuple[tuple[str, str], ...]:
    mapping_path = base_dir / "entity_name_to_ids.tsv"
    if not mapping_path.exists():
        return ()

    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [
            column for column in NAME_MAPPING_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            raise KGDataError(
                "Entity name mapping file "
                f"{mapping_path} is missing required columns: {', '.join(missing_columns)}."
            )
        rows: list[tuple[str, str]] = []
        for row in reader:
            raw_name = " ".join(str(row["name"]).split()).strip()
            raw_entity_id = str(row["id"]).strip()
            if not raw_name or not raw_entity_id:
                continue
            rows.append((raw_name, raw_entity_id))
    return tuple(rows)
