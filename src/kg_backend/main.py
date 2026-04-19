"""Local helpers for constructing and inspecting the KG backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kg_backend.backend import UncachedKGBackend
from kg_backend.errors import BackendUnavailableError


def _default_repo_data_path(repo_root: Path) -> Path:
    """Pick the default graph path inside the repository."""

    webqsp_path = repo_root / "datasets" / "WebQSP_KG"
    if webqsp_path.exists():
        return webqsp_path.resolve()
    fixture_path = repo_root / "src" / "kg_backend_tests" / "fixtures" / "tiny_kg.tsv"
    if fixture_path.exists():
        return fixture_path.resolve()
    raise BackendUnavailableError(
        "No KG data path configured. Set KG_BACKEND_DATA_PATH or pass --data-path."
    )


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve the graph data path from an argument, env var, or repo default."""

    if data_path is not None:
        return Path(data_path).expanduser().resolve()
    env_path = os.getenv("KG_BACKEND_DATA_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    return _default_repo_data_path(repo_root)


def build_backend(data_path: str | Path | None = None) -> UncachedKGBackend:
    """Construct the uncached backend from a resolved local graph path."""

    return UncachedKGBackend.from_data_path(resolve_data_path(data_path))


def main() -> None:
    """Print backend statistics for a local graph."""

    parser = argparse.ArgumentParser(description="Inspect a local uncached KG backend.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Graph directory or triples file.",
    )
    args = parser.parse_args()
    backend = build_backend(args.data_path)
    print(json.dumps(backend.stats().model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
