# GraphRAG SDK 2.0 — Ingestion: Resolution Strategies

from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
    ExactMatchResolution,
)
from graphrag_sdk.ingestion.resolution_strategies.semantic_resolution import (
    SemanticResolution,
)

__all__ = [
    "DescriptionMergeResolution",
    "ExactMatchResolution",
    "ResolutionStrategy",
    "SemanticResolution",
]
