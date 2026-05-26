# GraphRAG SDK — Core Foundation
# Stable contracts: models, providers, connection, context, exceptions.

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import GraphRAGError
from graphrag_sdk.core.models import (
    DocumentInfo,
    Entity,
    GraphNode,
    GraphRelationship,
    Relation,
    RetrieverResult,
    RetrieverResultItem,
    TextChunk,
)
from graphrag_sdk.core.providers import (
    Embedder,
    LiteLLM,
    LiteLLMEmbedder,
    LLMInterface,
    OpenRouterEmbedder,
    OpenRouterLLM,
)

__all__ = [
    "Context",
    "DocumentInfo",
    "Embedder",
    "Entity",
    "GraphNode",
    "GraphRAGError",
    "GraphRelationship",
    "LLMInterface",
    "LiteLLM",
    "LiteLLMEmbedder",
    "OpenRouterEmbedder",
    "OpenRouterLLM",
    "Relation",
    "RetrieverResult",
    "RetrieverResultItem",
    "TextChunk",
]


# ── Deprecation aliases ──────────────────────────────────────────
#
# Older names from before the v1.2.x ontology rename. Delegates to
# core.models.__getattr__ which emits the DeprecationWarning.

_LEGACY_CORE_ALIASES = {"EntityType", "RelationType", "PropertyType", "GraphSchema"}


def __getattr__(name: str):
    if name in _LEGACY_CORE_ALIASES:
        from graphrag_sdk.core import models as _models

        return getattr(_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
