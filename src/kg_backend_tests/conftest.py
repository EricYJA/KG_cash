"""Shared test fixtures for the KG backend package."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_backend.backend import UncachedKGBackend


@pytest.fixture(scope="session")
def tiny_kg_path() -> Path:
    """Return the path to the shared tiny KG fixture."""

    return (Path(__file__).parent / "fixtures" / "tiny_kg.tsv").resolve()


@pytest.fixture()
def backend(tiny_kg_path: Path) -> UncachedKGBackend:
    """Construct a backend over the shared tiny KG fixture."""

    return UncachedKGBackend.from_data_path(tiny_kg_path)
