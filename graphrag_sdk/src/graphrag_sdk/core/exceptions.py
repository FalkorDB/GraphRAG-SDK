# GraphRAG SDK — Core: Exception Hierarchy
# Centralized exceptions for the entire SDK.

from __future__ import annotations


class GraphRAGError(Exception):
    """Base exception for all GraphRAG SDK errors."""

    pass


class LatencyBudgetExceededError(GraphRAGError):
    """Raised when an operation cannot start within the remaining latency budget."""

    pass


# ── Provider Errors ──────────────────────────────────────────────


class LLMError(GraphRAGError):
    """Raised when an LLM provider call fails."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM provider call exceeds its configured timeout."""

    pass


class EmbeddingError(GraphRAGError):
    """Raised when an embedding provider call fails."""

    pass


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when an embedding provider call exceeds its configured timeout."""

    pass


# ── Ingestion Errors ─────────────────────────────────────────────


class IngestionError(GraphRAGError):
    """Base for all ingestion pipeline errors."""

    pass


class LoaderError(IngestionError):
    """Raised when a data loader fails."""

    pass


class ChunkingError(IngestionError):
    """Raised when a chunking strategy fails."""

    pass


class ExtractionError(IngestionError):
    """Raised when entity/relationship extraction fails."""

    pass


class ResolutionError(IngestionError):
    """Raised when entity resolution fails."""

    pass


# ── Retrieval Errors ─────────────────────────────────────────────


class RetrieverError(GraphRAGError):
    """Raised during retrieval operations."""

    pass


# ── Storage Errors ───────────────────────────────────────────────


class DatabaseError(GraphRAGError):
    """Raised for FalkorDB driver-level failures."""

    pass


class IndexError_(GraphRAGError):
    """Raised when index creation/management fails.

    Named with trailing underscore to avoid shadowing built-in IndexError.
    """

    pass


# ── Document Lifecycle Errors ────────────────────────────────────


class DocumentNotFoundError(GraphRAGError):
    """Raised when an operation references a Document node id that does
    not exist in the graph (e.g. ``update`` or ``delete_document`` with
    an unknown id and ``if_missing="error"``).
    """

    pass


# ── Schema Errors ────────────────────────────────────────────────


class SchemaValidationError(GraphRAGError):
    """Raised when graph schema validation fails."""

    pass


# ── Read-Only Surface Errors ─────────────────────────────────────


class ReadOnlyViolation(GraphRAGError):
    """Raised when a write is attempted through a read-only surface.

    Used by :mod:`graphrag_sdk.tools` both for guarded ``cypher_read``
    queries containing write clauses and for ``remember``/``flush``
    calls on a toolkit constructed with ``read_only=True``.

    Attributes:
        offending_token: The specific rejected token (e.g. ``"MERGE"``,
            ``"CALL apoc.load.json"``), or None for non-query violations.
    """

    def __init__(self, message: str, *, offending_token: str | None = None) -> None:
        super().__init__(message)
        self.offending_token = offending_token


# ── Configuration Errors ─────────────────────────────────────────


class ConfigError(GraphRAGError):
    """Raised when SDK configuration is invalid."""

    pass
