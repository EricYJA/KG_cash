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
    output_dir: str
    splits: tuple[str, ...]
    examples: int
    tuple_mentions: int
    unique_triples: int
    unique_entities: int
    unique_relations: int
    per_split: dict[str, SplitSummary]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the summary."""

        return {
            "source_dir": self.source_dir,
            "output_dir": self.output_dir,
            "splits": list(self.splits),
            "examples": self.examples,
            "tuple_mentions": self.tuple_mentions,
            "unique_triples": self.unique_triples,
            "unique_entities": self.unique_entities,
            "unique_relations": self.unique_relations,
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
    head, relation, tail = (str(item).strip() if item is not None else "" for item in raw_triple)
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
) -> tuple[tuple[tuple[str, str, str], ...], tuple[str, ...], tuple[str, ...], dict[str, SplitSummary], int, int]:
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
    entity_ids = tuple(sorted({head for head, _, _ in sorted_triples} | {tail for _, _, tail in sorted_triples}))
    relation_ids = tuple(sorted({relation for _, relation, _ in sorted_triples}))

    return (
        sorted_triples,
        entity_ids,
        relation_ids,
        per_split,
        total_examples,
        total_tuple_mentions,
    )


def write_tsv(path: Path, header: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> None:
    """Write rows to a UTF-8 TSV file with a header."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def build_webqsp_subkg(
    *,
    source_dir: Path,
    output_dir: Path,
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

    write_tsv(output_dir / "triples.tsv", ("head", "relation", "tail"), triples)
    write_tsv(output_dir / "entities.tsv", ("id", "label"), ((entity_id, entity_id) for entity_id in entity_ids))
    write_tsv(
        output_dir / "relations.tsv",
        ("id", "label"),
        ((relation_id, relation_id) for relation_id in relation_ids),
    )

    summary = BuildSummary(
        source_dir=str(source_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        splits=tuple(splits),
        examples=total_examples,
        tuple_mentions=total_tuple_mentions,
        unique_triples=len(triples),
        unique_entities=len(entity_ids),
        unique_relations=len(relation_ids),
        per_split=per_split,
    )
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        splits=tuple(args.splits),
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
