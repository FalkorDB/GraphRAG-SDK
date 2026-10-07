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


class DatabaseUnavailableError(DatabaseError):
    """Raised when FalkorDB cannot be reached at all.

    Distinct from a plain :class:`DatabaseError`, which means the server
    answered and *rejected* the query. Callers that fall back to a slower
    path on cache/read failures need that distinction: a rejected query
    leaves the rest of the pipeline perfectly usable, while an unreachable
    server will fail the write phase no matter how much work is done first.

    Subclasses ``DatabaseError``, so existing ``except DatabaseError``
    handlers keep working unchanged.
    """

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


# ── Configuration Errors ─────────────────────────────────────────


class ConfigError(GraphRAGError):
    """Raised when SDK configuration is invalid."""

    pass
