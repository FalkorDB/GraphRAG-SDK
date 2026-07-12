"""Agentic GraphRAG toolkit — framework-neutral agent surface over GraphRAG.

Canonical import: ``from graphrag_sdk.tools import GraphRAGToolkit``.
"""

from __future__ import annotations

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.tools.models import (
    AnswerResult,
    ChunkRef,
    Citation,
    CypherResult,
    DocumentRef,
    EntityCard,
    EntityResult,
    EntityTypeInfo,
    RelationTriple,
    RelationTypeInfo,
    RememberResult,
    SchemaResult,
    SearchResult,
    ToolResult,
)
from graphrag_sdk.tools.specs import ToolSpec

__all__ = [
    "AnswerResult",
    "ChunkRef",
    "Citation",
    "CypherResult",
    "DocumentRef",
    "EntityCard",
    "EntityResult",
    "EntityTypeInfo",
    "ReadOnlyViolation",
    "RelationTriple",
    "RelationTypeInfo",
    "RememberResult",
    "SchemaResult",
    "SearchResult",
    "ToolResult",
    "ToolSpec",
]
