"""Tests for backend startup helpers."""

from __future__ import annotations

from pathlib import Path

from kg_backend.main import _default_repo_data_path


def test_default_repo_data_path_prefers_webqsp_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets" / "WebQSP_KG"
    dataset_dir.mkdir(parents=True)
    fixture_path = tmp_path / "src" / "kg_backend_tests" / "fixtures" / "tiny_kg.tsv"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text("head\trelation\ttail\nalice\tlikes\tbob\n", encoding="utf-8")

    assert _default_repo_data_path(tmp_path) == dataset_dir.resolve()


def test_default_repo_data_path_falls_back_to_fixture(tmp_path: Path) -> None:
    fixture_path = tmp_path / "src" / "kg_backend_tests" / "fixtures" / "tiny_kg.tsv"
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_text("head\trelation\ttail\nalice\tlikes\tbob\n", encoding="utf-8")

    assert _default_repo_data_path(tmp_path) == fixture_path.resolve()
