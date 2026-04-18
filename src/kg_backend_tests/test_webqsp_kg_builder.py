"""Tests for the WebQSP dataset-specific KG builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_builder_module():
    """Load the WebQSP KG builder module from the datasets directory."""

    script_path = (
        Path(__file__).resolve().parents[2] / "datasets" / "WebQSP_KG" / "build_webqsp_subkg.py"
    )
    spec = importlib.util.spec_from_file_location("build_webqsp_subkg", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small JSONL file for builder tests."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_builder_writes_deduplicated_backend_files(tmp_path: Path) -> None:
    builder = load_builder_module()
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()

    write_jsonl(
        source_dir / "train_simple.jsonl",
        [
            {
                "id": "q1",
                "subgraph": {
                    "tuples": [
                        ["m.a", "r.x", "m.b"],
                        ["m.a", "r.x", "m.b"],
                        ["m.b", "r.y", "2010"],
                    ]
                },
            },
            {
                "id": "q2",
                "subgraph": {"tuples": [["m.c", "r.x", "m.b"]]},
            },
        ],
    )
    write_jsonl(
        source_dir / "dev_simple.jsonl",
        [{"id": "q3", "subgraph": {"tuples": [["m.c", "r.z", "literal"]]}}],
    )
    write_jsonl(
        source_dir / "test_simple.jsonl",
        [{"id": "q4", "subgraph": {"tuples": [["m.a", "r.x", "m.b"]]}}],
    )

    summary = builder.build_webqsp_subkg(source_dir=source_dir, output_dir=output_dir)

    assert summary.examples == 4
    assert summary.tuple_mentions == 6
    assert summary.unique_triples == 4
    assert summary.unique_entities == 5
    assert summary.unique_relations == 3

    assert (output_dir / "triples.tsv").read_text(encoding="utf-8").splitlines() == [
        "head\trelation\ttail",
        "m.a\tr.x\tm.b",
        "m.b\tr.y\t2010",
        "m.c\tr.x\tm.b",
        "m.c\tr.z\tliteral",
    ]
    assert (output_dir / "entities.tsv").read_text(encoding="utf-8").splitlines() == [
        "id\tlabel",
        "2010\t2010",
        "literal\tliteral",
        "m.a\tm.a",
        "m.b\tm.b",
        "m.c\tm.c",
    ]
    assert (output_dir / "relations.tsv").read_text(encoding="utf-8").splitlines() == [
        "id\tlabel",
        "r.x\tr.x",
        "r.y\tr.y",
        "r.z\tr.z",
    ]

    stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["examples"] == 4
    assert stats["unique_triples"] == 4
