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
    """Raised when FalkorDB could not run the query now.

    The server is unreachable, the circuit breaker is open, or a temporary
    condition (connection loss or socket timeout, full query queue,
    ``BUSY``/``LOADING``/``NOREPLICAS``/``MISCONF``, cluster redirect or
    failover, read-only replica, out of memory) outlasted every retry.
    Distinct from a plain :class:`DatabaseError`, which means the server
    answered and *rejected* the query, or answered that it ran past its
    timeout. Callers that fall back to a slower path on cache/read failures
    need that distinction: a rejected query leaves the rest of the pipeline
    perfectly usable, while an unreachable server will fail the write phase
    no matter how much work is done first.

    Subclasses ``DatabaseError``, so existing ``except DatabaseError``
    handlers keep working unchanged.
    """

    pass


class FalkorDBVersionError(DatabaseError):
    """Raised when the FalkorDB server is older than the SDK supports.

    Checked once per connection, before its first query, so nothing is
    written to a server the SDK would misuse. Subclasses ``DatabaseError``.
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
