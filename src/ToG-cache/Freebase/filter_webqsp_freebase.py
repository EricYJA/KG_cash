#!/usr/bin/env python3
"""Create a WebQSP-only Freebase RDF file from FilterFreebase."""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


ENTITY_RE = re.compile(
    r"(?:http://rdf\.freebase\.com/ns/|(?:^|[^A-Za-z0-9_])(?:ns:|:))"
    r"([mg]\.[A-Za-z0-9_]+)"
)
URI_ENTITY_RE = re.compile(r"^<http://rdf\.freebase\.com/ns/([mg]\.[A-Za-z0-9_]+)>$")


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return path.open(mode, encoding="utf-8", errors="replace")


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def collect_webqsp_entities(paths: list[Path]) -> set[str]:
    entities: set[str] = set()
    for path in paths:
        with open_text(path) as handle:
            data = json.load(handle)
        before = len(entities)
        for text in iter_strings(data):
            for match in ENTITY_RE.finditer(text):
                entities.add(match.group(1))
        print(f"{path}: +{len(entities) - before:,} entities", file=sys.stderr)
    return entities


def rdf_entity(term: str) -> str | None:
    match = URI_ENTITY_RE.match(term)
    if match:
        return match.group(1)
    return None


def keep_line(line: str, entities: set[str]) -> bool:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return False
    return rdf_entity(parts[0]) in entities or rdf_entity(parts[2]) in entities


def filter_freebase(input_path: Path, output_path: Path, entities: set[str]) -> None:
    kept = 0
    read = 0
    with open_text(input_path) as source, open_text(output_path, "wt") as sink:
        for line in source:
            read += 1
            if keep_line(line, entities):
                sink.write(line)
                kept += 1
            if read % 1_000_000 == 0:
                print(f"read={read:,} kept={kept:,}", file=sys.stderr)
    print(f"done: read={read:,} kept={kept:,}", file=sys.stderr)


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    webqsp_paths = [
        root / "datasets/WebQSP/data/WebQSP.train.json",
        root / "datasets/WebQSP/data/WebQSP.test.json",
    ]
    input_path = here / "FilterFreebase"
    output_path = here / "WebQSP_FilterFreebase"

    missing = [path for path in [*webqsp_paths, input_path] if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        return 1

    entities = collect_webqsp_entities(webqsp_paths)
    print(f"total WebQSP entities: {len(entities):,}", file=sys.stderr)
    filter_freebase(input_path, output_path, entities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
