# GraphRAG SDK 2.0 — Core: Exception Hierarchy
# Centralized exceptions for the entire SDK.

from __future__ import annotations


class GraphRAGError(Exception):
    """Base exception for all GraphRAG SDK errors."""

    pass


# ── Provider Errors ──────────────────────────────────────────────


class LLMError(GraphRAGError):
    """Raised when an LLM provider call fails."""

    pass


class EmbeddingError(GraphRAGError):
    """Raised when an embedding provider call fails."""

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


# ── Schema Errors ────────────────────────────────────────────────


class SchemaValidationError(GraphRAGError):
    """Raised when graph schema validation fails."""

    pass


# ── Configuration Errors ─────────────────────────────────────────


class ConfigError(GraphRAGError):
    """Raised when SDK configuration is invalid."""

    pass
