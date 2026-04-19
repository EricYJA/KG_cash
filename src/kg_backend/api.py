"""FastAPI application for the uncached KG backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from kg_backend.backend import UncachedKGBackend
from kg_backend.errors import BackendUnavailableError, KGBackendError
from kg_backend.main import resolve_data_path
from kg_backend.types import GraphStats, KGQuery


def create_app(data_path: str | Path | None = None) -> FastAPI:
    """Create a FastAPI app bound to one uncached backend instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        try:
            resolved_data_path = resolve_data_path(data_path)
            app.state.backend = UncachedKGBackend.from_data_path(resolved_data_path)
            app.state.backend_error = None
        except KGBackendError as exc:
            app.state.backend = None
            app.state.backend_error = exc
        yield

    app = FastAPI(title="KG Backend", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(KGBackendError)
    async def handle_backend_error(_: Request, exc: KGBackendError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.get("/health")
    async def health() -> dict[str, Any]:
        backend = getattr(app.state, "backend", None)
        error = getattr(app.state, "backend_error", None)
        return {
            "status": "ok" if backend is not None else "degraded",
            "backend_loaded": backend is not None,
            "detail": None if error is None else str(error),
        }

    @app.get("/stats", response_model=GraphStats)
    async def stats() -> GraphStats:
        return _get_backend(app).stats()

    @app.post("/query")
    async def query(query: KGQuery) -> dict[str, object]:
        return _get_backend(app).execute(query)

    return app


def _get_backend(app: FastAPI) -> UncachedKGBackend:
    backend = getattr(app.state, "backend", None)
    if isinstance(backend, UncachedKGBackend):
        return backend
    if backend is None:
        error = getattr(app.state, "backend_error", None)
        if isinstance(error, KGBackendError):
            raise BackendUnavailableError(str(error))
        raise BackendUnavailableError("Backend is not available.")
    raise BackendUnavailableError("Backend is not available.")


app = create_app()
