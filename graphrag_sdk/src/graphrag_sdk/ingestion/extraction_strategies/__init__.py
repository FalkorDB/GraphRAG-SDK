# GraphRAG SDK 2.0 — Ingestion: Extraction Strategies

from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.coref_resolvers import (
    CorefResolver,
    FastCorefResolver,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
    EntityExtractor,
)
from graphrag_sdk.ingestion.extraction_strategies.hybrid_extraction import (
    HybridExtraction,
)

__all__ = [
    "ExtractionStrategy",
    # Main strategy
    "HybridExtraction",
    # Pluggable entity extractor
    "EntityExtractor",
    # Pluggable coreference resolvers
    "CorefResolver",
    "FastCorefResolver",
]
