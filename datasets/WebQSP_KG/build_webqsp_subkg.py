#!/usr/bin/env python3
"""Build a WebQSP dataset-specific subKG for src/kg_backend."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SPLITS = ("train", "dev", "test")
DEFAULT_WEBQSP_JSON_FILES = ("WebQSP.train.json", "WebQSP.test.json")
NAME_RELATION_ID = "type.object.name"
SOURCE_PRIORITY = {
    "type.object.name": 0,
    "webqsp_topic_entity": 1,
    "webqsp_answer_entity": 2,
    "webqsp_constraint_entity": 3,
    "epr_answer_text": 4,
}


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Summary statistics for one WebQSP split."""

    examples: int
    tuple_mentions: int
    unique_triples: int


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Summary statistics for one built WebQSP subKG."""

    source_dir: str
    webqsp_data_dir: str | None
    output_dir: str
    splits: tuple[str, ...]
    examples: int
    tuple_mentions: int
    unique_triples: int
    unique_entities: int
    unique_relations: int
    named_entities: int
    entity_name_pairs: int
    per_split: dict[str, SplitSummary]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the summary."""

        return {
            "source_dir": self.source_dir,
            "webqsp_data_dir": self.webqsp_data_dir,
            "output_dir": self.output_dir,
            "splits": list(self.splits),
            "examples": self.examples,
            "tuple_mentions": self.tuple_mentions,
            "unique_triples": self.unique_triples,
            "unique_entities": self.unique_entities,
            "unique_relations": self.unique_relations,
            "named_entities": self.named_entities,
            "entity_name_pairs": self.entity_name_pairs,
            "per_split": {
                split: {
                    "examples": summary.examples,
                    "tuple_mentions": summary.tuple_mentions,
                    "unique_triples": summary.unique_triples,
                }
                for split, summary in self.per_split.items()
            },
            "notes": [
                "This KG is the union of question-specific extracted subgraphs from EPR-KGQA.",
                "Literal values are preserved as string node ids so the graph stays loadable by src/kg_backend.",
                "entities.tsv uses recovered human-readable labels when available and falls back to raw ids otherwise.",
                "entity_name_to_ids.tsv stores name-to-id pairs for reverse lookup because names are not unique.",
            ],
        }


def repo_root() -> Path:
    """Return the repository root from this script location."""

    return Path(__file__).resolve().parents[2]


def default_source_dir() -> Path:
    """Return the default EPR-KGQA WebQSP source directory."""

    return repo_root() / "ref_KG_projects" / "EPR-KGQA" / "data" / "dataset" / "WebQSP"


def default_output_dir() -> Path:
    """Return the default output directory for the built WebQSP subKG."""

    return Path(__file__).resolve().parent


def default_webqsp_data_dir() -> Path:
    """Return the default directory for the original WebQSP JSON files."""

    return repo_root() / "datasets" / "WebQSP" / "data"


def normalize_triple(
    raw_triple: object,
    *,
    source_file: Path,
    line_number: int,
) -> tuple[str, str, str]:
    """Normalize one raw JSON tuple into a validated string triple."""

    if not isinstance(raw_triple, list) or len(raw_triple) != 3:
        raise ValueError(
            f"Expected a three-item tuple in {source_file}:{line_number}, got {raw_triple!r}."
        )
    head, relation, tail = (
        str(item).strip() if item is not None else "" for item in raw_triple
    )
    if not head or not relation or not tail:
        raise ValueError(
            f"Encountered an empty triple field in {source_file}:{line_number}: {raw_triple!r}."
        )
    return head, relation, tail


def extract_example_triples(
    example: dict[str, Any],
    *,
    source_file: Path,
    line_number: int,
) -> list[tuple[str, str, str]]:
    """Extract normalized triples from one WebQSP example record."""

    subgraph = example.get("subgraph")
    if not isinstance(subgraph, dict):
        raise ValueError(f"Missing subgraph in {source_file}:{line_number}.")
    raw_tuples = subgraph.get("tuples")
    if not isinstance(raw_tuples, list):
        raise ValueError(f"Missing subgraph.tuples in {source_file}:{line_number}.")
    return [
        normalize_triple(raw_triple, source_file=source_file, line_number=line_number)
        for raw_triple in raw_tuples
    ]


def collect_webqsp_subkg(
    source_dir: Path,
    splits: Sequence[str],
) -> tuple[
    tuple[tuple[str, str, str], ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, SplitSummary],
    int,
    int,
]:
    """Collect deduplicated triples, entities, and relations from the requested splits."""

    all_triples: set[tuple[str, str, str]] = set()
    per_split: dict[str, SplitSummary] = {}
    total_examples = 0
    total_tuple_mentions = 0

    for split in splits:
        split_file = source_dir / f"{split}_simple.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        split_unique: set[tuple[str, str, str]] = set()
        split_examples = 0
        split_tuple_mentions = 0

        with split_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    example = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {split_file}:{line_number}: {exc.msg}"
                    ) from exc
                example_triples = extract_example_triples(
                    example,
                    source_file=split_file,
                    line_number=line_number,
                )
                split_examples += 1
                split_tuple_mentions += len(example_triples)
                split_unique.update(example_triples)
                all_triples.update(example_triples)

        total_examples += split_examples
        total_tuple_mentions += split_tuple_mentions
        per_split[split] = SplitSummary(
            examples=split_examples,
            tuple_mentions=split_tuple_mentions,
            unique_triples=len(split_unique),
        )

    sorted_triples = tuple(sorted(all_triples))
    entity_ids = tuple(
        sorted(
            {head for head, _, _ in sorted_triples}
            | {tail for _, _, tail in sorted_triples}
        )
    )
    relation_ids = tuple(sorted({relation for _, relation, _ in sorted_triples}))

    return (
        sorted_triples,
        entity_ids,
        relation_ids,
        per_split,
        total_examples,
        total_tuple_mentions,
    )


def normalize_name(raw_name: object) -> str:
    """Normalize one candidate entity name into a stable single-line string."""

    if raw_name is None:
        return ""
    return " ".join(str(raw_name).split()).strip()


def add_entity_name(
    entity_names: dict[str, dict[str, set[str]]],
    *,
    entity_id: object,
    name: object,
    source: str,
) -> None:
    """Record one entity-id to human-readable-name mapping candidate."""

    normalized_entity_id = str(entity_id).strip() if entity_id is not None else ""
    normalized_name = normalize_name(name)
    if not normalized_entity_id or not normalized_name:
        return
    entity_names.setdefault(normalized_entity_id, {}).setdefault(
        normalized_name, set()
    ).add(source)


def collect_names_from_epr_sources(
    source_dir: Path,
    splits: Sequence[str],
) -> dict[str, dict[str, set[str]]]:
    """Collect entity names from EPR WebQSP traces."""

    entity_names: dict[str, dict[str, set[str]]] = {}
    for split in splits:
        split_file = source_dir / f"{split}_simple.jsonl"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        with split_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    example = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {split_file}:{line_number}: {exc.msg}"
                    ) from exc

                for head, relation, tail in extract_example_triples(
                    example,
                    source_file=split_file,
                    line_number=line_number,
                ):
                    if relation == NAME_RELATION_ID:
                        add_entity_name(
                            entity_names,
                            entity_id=head,
                            name=tail,
                            source="type.object.name",
                        )

                raw_answers = example.get("answers")
                if not isinstance(raw_answers, list):
                    continue
                for answer in raw_answers:
                    if not isinstance(answer, dict):
                        continue
                    add_entity_name(
                        entity_names,
                        entity_id=answer.get("kb_id"),
                        name=answer.get("text"),
                        source="epr_answer_text",
                    )
    return entity_names


def collect_names_from_webqsp_json(
    webqsp_data_dir: Path | None,
) -> dict[str, dict[str, set[str]]]:
    """Collect entity names from the original WebQSP train/test JSON files."""

    entity_names: dict[str, dict[str, set[str]]] = {}
    if webqsp_data_dir is None or not webqsp_data_dir.exists():
        return entity_names

    for filename in DEFAULT_WEBQSP_JSON_FILES:
        json_path = webqsp_data_dir / filename
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw_questions = payload.get("Questions")
        if not isinstance(raw_questions, list):
            raise ValueError(f"Missing Questions list in {json_path}.")
        for raw_question in raw_questions:
            if not isinstance(raw_question, dict):
                continue
            raw_parses = raw_question.get("Parses")
            if not isinstance(raw_parses, list):
                continue
            for raw_parse in raw_parses:
                if not isinstance(raw_parse, dict):
                    continue
                add_entity_name(
                    entity_names,
                    entity_id=raw_parse.get("TopicEntityMid"),
                    name=raw_parse.get("TopicEntityName"),
                    source="webqsp_topic_entity",
                )
                raw_constraints = raw_parse.get("Constraints")
                if isinstance(raw_constraints, list):
                    for raw_constraint in raw_constraints:
                        if not isinstance(raw_constraint, dict):
                            continue
                        if raw_constraint.get("ArgumentType") != "Entity":
                            continue
                        add_entity_name(
                            entity_names,
                            entity_id=raw_constraint.get("Argument"),
                            name=raw_constraint.get("EntityName"),
                            source="webqsp_constraint_entity",
                        )
                raw_answers = raw_parse.get("Answers")
                if isinstance(raw_answers, list):
                    for raw_answer in raw_answers:
                        if not isinstance(raw_answer, dict):
                            continue
                        if raw_answer.get("AnswerType") != "Entity":
                            continue
                        add_entity_name(
                            entity_names,
                            entity_id=raw_answer.get("AnswerArgument"),
                            name=raw_answer.get("EntityName"),
                            source="webqsp_answer_entity",
                        )
    return entity_names


def merge_entity_name_maps(
    *entity_name_maps: dict[str, dict[str, set[str]]],
) -> dict[str, dict[str, set[str]]]:
    """Merge multiple entity-name maps into one deterministic mapping."""

    merged: dict[str, dict[str, set[str]]] = {}
    for entity_name_map in entity_name_maps:
        for entity_id, names in entity_name_map.items():
            target = merged.setdefault(entity_id, {})
            for name, sources in names.items():
                target.setdefault(name, set()).update(sources)
    return merged


def choose_canonical_name(name_sources: dict[str, set[str]]) -> str:
    """Choose one stable canonical label for an entity from known names."""

    return min(
        name_sources.items(),
        key=lambda item: (
            min(SOURCE_PRIORITY.get(source, 999) for source in item[1]),
            len(item[0]),
            item[0].casefold(),
            item[0],
        ),
    )[0]


def build_entity_metadata(
    entity_ids: Sequence[str],
    entity_name_candidates: dict[str, dict[str, set[str]]],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...], int]:
    """Build canonical entity labels and reverse name-to-id rows."""

    canonical_rows: list[tuple[str, str]] = []
    reverse_rows: list[tuple[str, str]] = []
    named_entities = 0

    for entity_id in entity_ids:
        name_sources = entity_name_candidates.get(entity_id, {})
        if name_sources:
            canonical_label = choose_canonical_name(name_sources)
            named_entities += 1
            for name in sorted(
                name_sources, key=lambda candidate: (candidate.casefold(), candidate)
            ):
                reverse_rows.append((name, entity_id))
        else:
            canonical_label = entity_id
        canonical_rows.append((entity_id, canonical_label))

    filtered_reverse_rows = tuple(
        sorted(reverse_rows, key=lambda row: (row[0].casefold(), row[0], row[1]))
    )
    return tuple(canonical_rows), filtered_reverse_rows, named_entities


def write_tsv(
    path: Path, header: tuple[str, ...], rows: Iterable[tuple[str, ...]]
) -> None:
    """Write rows to a UTF-8 TSV file with a header."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def build_webqsp_subkg(
    *,
    source_dir: Path,
    output_dir: Path,
    webqsp_data_dir: Path | None = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> BuildSummary:
    """Build the WebQSP dataset-specific subKG under the requested output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (
        triples,
        entity_ids,
        relation_ids,
        per_split,
        total_examples,
        total_tuple_mentions,
    ) = collect_webqsp_subkg(source_dir, tuple(splits))

    entity_name_candidates = merge_entity_name_maps(
        collect_names_from_epr_sources(source_dir, tuple(splits)),
        collect_names_from_webqsp_json(webqsp_data_dir),
    )
    entity_rows, entity_name_rows, named_entities = build_entity_metadata(
        entity_ids,
        entity_name_candidates,
    )

    write_tsv(output_dir / "triples.tsv", ("head", "relation", "tail"), triples)
    write_tsv(output_dir / "entities.tsv", ("id", "label"), entity_rows)
    write_tsv(
        output_dir / "relations.tsv",
        ("id", "label"),
        ((relation_id, relation_id) for relation_id in relation_ids),
    )
    write_tsv(output_dir / "entity_name_to_ids.tsv", ("name", "id"), entity_name_rows)

    summary = BuildSummary(
        source_dir=str(source_dir.resolve()),
        webqsp_data_dir=None
        if webqsp_data_dir is None
        else str(webqsp_data_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        splits=tuple(splits),
        examples=total_examples,
        tuple_mentions=total_tuple_mentions,
        unique_triples=len(triples),
        unique_entities=len(entity_ids),
        unique_relations=len(relation_ids),
        named_entities=named_entities,
        entity_name_pairs=len(entity_name_rows),
        per_split=per_split,
    )
    stats_path = output_dir / "stats.json"
    stats_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the WebQSP subKG builder."""

    parser = argparse.ArgumentParser(
        description="Build a WebQSP dataset-specific subKG from EPR-KGQA WebQSP simple jsonl files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_dir(),
        help="Directory containing train_simple.jsonl, dev_simple.jsonl, and test_simple.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where triples.tsv and metadata files will be written.",
    )
    parser.add_argument(
        "--webqsp-data-dir",
        type=Path,
        default=default_webqsp_data_dir(),
        help="Optional directory containing WebQSP.train.json and WebQSP.test.json for extra entity names.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="WebQSP splits to include when constructing the dataset-specific subKG.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the WebQSP dataset-specific subKG and print a compact summary."""

    args = parse_args()
    summary = build_webqsp_subkg(
        source_dir=args.source_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        webqsp_data_dir=args.webqsp_data_dir.expanduser().resolve(),
        splits=tuple(args.splits),
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
