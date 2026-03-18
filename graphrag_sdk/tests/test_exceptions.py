"""Tests for core/exceptions.py — exception hierarchy."""
from __future__ import annotations

import pytest

from graphrag_sdk.core.exceptions import (
    ChunkingError,
    ConfigError,
    DatabaseError,
    EmbeddingError,
    ExtractionError,
    GraphRAGError,
    IndexError_,
    IngestionError,
    LLMError,
    LoaderError,
    ResolutionError,
    RetrieverError,
    SchemaValidationError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        with pytest.raises(GraphRAGError):
            raise GraphRAGError("base error")

    def test_llm_error_is_graphrag_error(self):
        with pytest.raises(GraphRAGError):
            raise LLMError("llm failed")

    def test_embedding_error_is_graphrag_error(self):
        with pytest.raises(GraphRAGError):
            raise EmbeddingError("embedding failed")

    def test_ingestion_error_hierarchy(self):
        assert issubclass(IngestionError, GraphRAGError)
        assert issubclass(LoaderError, IngestionError)
        assert issubclass(ChunkingError, IngestionError)
        assert issubclass(ExtractionError, IngestionError)
        assert issubclass(ResolutionError, IngestionError)

    def test_loader_error_caught_as_ingestion_error(self):
        with pytest.raises(IngestionError):
            raise LoaderError("file not found")

    def test_retriever_error(self):
        with pytest.raises(GraphRAGError):
            raise RetrieverError("search failed")

    def test_database_error(self):
        with pytest.raises(GraphRAGError):
            raise DatabaseError("connection lost")

    def test_index_error(self):
        with pytest.raises(GraphRAGError):
            raise IndexError_("index creation failed")

    def test_schema_validation_error(self):
        with pytest.raises(GraphRAGError):
            raise SchemaValidationError("invalid schema")

    def test_config_error(self):
        with pytest.raises(GraphRAGError):
            raise ConfigError("bad config")

    def test_exception_messages(self):
        err = LoaderError("file missing: test.pdf")
        assert str(err) == "file missing: test.pdf"
        assert isinstance(err, IngestionError)
        assert isinstance(err, GraphRAGError)
        assert isinstance(err, Exception)
