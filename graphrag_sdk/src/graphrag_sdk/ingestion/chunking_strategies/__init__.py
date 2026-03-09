# GraphRAG SDK 2.0 — Ingestion: Chunking Strategies

from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking
from graphrag_sdk.ingestion.chunking_strategies.contextual_chunking import ContextualChunking

__all__ = [
    "ChunkingStrategy",
    "FixedSizeChunking",
    "SentenceTokenCapChunking",
    "ContextualChunking",
]

# Llama strategies are optional — install graphrag-sdk[llama] to use them.
# Import directly: from graphrag_sdk.ingestion.chunking_strategies.llama_sentence import LlamaSentenceChunking
try:
    from graphrag_sdk.ingestion.chunking_strategies.llama_sentence import LlamaSentenceChunking
    from graphrag_sdk.ingestion.chunking_strategies.llama_semantic import LlamaSemanticChunking
    from graphrag_sdk.ingestion.chunking_strategies.llama_semantic_double import LlamaSemanticDoubleChunking
    from graphrag_sdk.ingestion.chunking_strategies.llama_topic import LlamaTopicChunking
except ImportError:
    # Optional llama-based strategies are unavailable when the 'llama' extra is not installed.
    pass
else:
    __all__.extend([
        "LlamaSentenceChunking",
        "LlamaSemanticChunking",
        "LlamaSemanticDoubleChunking",
        "LlamaTopicChunking",
    ])
