"""Typed exceptions used by the KG backend."""

from __future__ import annotations


class KGBackendError(Exception):
    """Base class for backend-specific errors."""

    status_code = 400
    error_code = "kg_backend_error"

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-serializable error payload."""
        return {"error": self.error_code, "detail": str(self)}


class KGDataError(KGBackendError):
    """Raised when graph files are missing or invalid."""

    error_code = "kg_data_error"


class MetadataValidationError(KGDataError):
    """Raised when optional metadata files disagree with the triples."""

    error_code = "metadata_validation_error"


class EntityNotFoundError(KGBackendError):
    """Raised when a query references an unknown entity identifier."""

    status_code = 404
    error_code = "entity_not_found"


class RelationNotFoundError(KGBackendError):
    """Raised when a query references an unknown relation identifier."""

    status_code = 404
    error_code = "relation_not_found"


class BackendUnavailableError(KGBackendError):
    """Raised when the API has no configured backend instance."""

    status_code = 503
    error_code = "backend_unavailable"
