# GraphRAG SDK 2.0 — API: GraphRAG Facade
# Pattern: Facade — single entry point that hides all internal wiring.
# Principle: Simplicity — two-line usage: init + query/ingest.

from __future__ import annotations

import logging
from typing import Any, Optional

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import GraphRAGError
from graphrag_sdk.core.models import (
    GraphSchema,
    IngestionResult,
    RagResult,
    RetrieverResult,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.schema_guided import SchemaGuidedExtraction
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader
from graphrag_sdk.ingestion.loaders.text_loader import TextLoader
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.local import LocalRetrieval
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


_RAG_PROMPT = (
    "Answer the user's question using the provided context.\n"
    "If the context doesn't contain enough information, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class GraphRAG:
    """The main user-facing class for GraphRAG operations.

    Provides two primary operations:
    - ``ingest()`` — build a knowledge graph from sources
    - ``query()`` — search the knowledge graph and generate answers

    Both use sensible defaults but allow full strategy customisation.

    Args:
        connection: FalkorDB connection (or ConnectionConfig to create one).
        llm: LLM provider for extraction and generation.
        embedder: Embedding provider for vector operations.
        schema: Optional graph schema for extraction constraints.
        retrieval_strategy: Default retrieval strategy (uses LocalRetrieval if None).

    Example::

        rag = GraphRAG(
            connection=ConnectionConfig(host="localhost", graph_name="my_graph"),
            llm=MyLLM(model_name="gpt-4o"),
            embedder=MyEmbedder(),
            schema=GraphSchema(entities=[...], relations=[...]),
        )

        # Ingest
        await rag.ingest("my_document.pdf")

        # Query
        result = await rag.query("What is the capital of France?")
        print(result.answer)
    """

    def __init__(
        self,
        connection: FalkorDBConnection | ConnectionConfig,
        llm: LLMInterface,
        embedder: Embedder,
        schema: GraphSchema | None = None,
        retrieval_strategy: RetrievalStrategy | None = None,
    ) -> None:
        # Connection
        if isinstance(connection, ConnectionConfig):
            self._conn = FalkorDBConnection(connection)
        else:
            self._conn = connection

        self.llm = llm
        self.embedder = embedder
        self.schema = schema or GraphSchema()

        # Storage layer
        self.graph_store = GraphStore(self._conn)
        self.vector_store = VectorStore(
            self._conn,
            embedder=self.embedder,
        )

        # Default retrieval strategy
        self._retrieval_strategy = retrieval_strategy or LocalRetrieval(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

    # ── Ingestion ────────────────────────────────────────────────

    async def ingest(
        self,
        source: str,
        *,
        text: str | None = None,
        loader: LoaderStrategy | None = None,
        chunker: ChunkingStrategy | None = None,
        extractor: ExtractionStrategy | None = None,
        resolver: ResolutionStrategy | None = None,
        ctx: Context | None = None,
    ) -> IngestionResult:
        """Build a knowledge graph from a source.

        Uses sensible defaults for any unspecified strategy:
        - Loader: auto-detected from file extension (PDF or text)
        - Chunker: FixedSizeChunking(chunk_size=1000)
        - Extractor: SchemaGuidedExtraction with configured LLM
        - Resolver: ExactMatchResolution

        Args:
            source: File path, URL, or identifier for the data source.
            text: Optional raw text (skips loader if provided).
            loader: Custom loader strategy.
            chunker: Custom chunking strategy.
            extractor: Custom extraction strategy.
            resolver: Custom resolution strategy.
            ctx: Execution context.

        Returns:
            IngestionResult with pipeline statistics.
        """
        if ctx is None:
            ctx = Context()

        # Auto-detect loader from file extension
        if loader is None and text is None:
            if source.lower().endswith(".pdf"):
                loader = PdfLoader()
            else:
                loader = TextLoader()

        pipeline = IngestionPipeline(
            loader=loader or TextLoader(),
            chunker=chunker or FixedSizeChunking(),
            extractor=extractor or SchemaGuidedExtraction(llm=self.llm),
            resolver=resolver or ExactMatchResolution(),
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            schema=self.schema,
        )

        return await pipeline.run(source, ctx, text=text)

    # ── Query ────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        *,
        strategy: RetrievalStrategy | None = None,
        reranker: RerankingStrategy | None = None,
        prompt_template: str | None = None,
        return_context: bool = False,
        ctx: Context | None = None,
    ) -> RagResult:
        """Query the knowledge graph and generate an answer.

        Flow: Retrieve context → (optional) Rerank → Generate answer.

        Args:
            question: The user's question.
            strategy: Override retrieval strategy (uses default if None).
            reranker: Optional reranking strategy to apply.
            prompt_template: Custom prompt template (must contain {context} and {question}).
            return_context: If True, include retriever results in output.
            ctx: Execution context.

        Returns:
            RagResult with the generated answer.
        """
        if ctx is None:
            ctx = Context()

        ctx.log(f"Query: {question[:80]}...")

        # Step 1: Retrieve
        retrieval = strategy or self._retrieval_strategy
        retriever_result = await retrieval.search(question, ctx)

        # Step 2: Rerank (optional)
        if reranker is not None:
            retriever_result = await reranker.rerank(question, retriever_result, ctx)

        # Step 3: Build context
        context_str = "\n---\n".join(item.content for item in retriever_result.items)

        # Step 4: Generate answer
        template = prompt_template or _RAG_PROMPT
        prompt = template.format(context=context_str, question=question)
        llm_response = await self.llm.ainvoke(prompt)

        result = RagResult(
            answer=llm_response.content,
            retriever_result=retriever_result if return_context else None,
            metadata={
                "model": self.llm.model_name,
                "num_context_items": len(retriever_result.items),
                "strategy": retrieval.__class__.__name__,
            },
        )

        ctx.log(f"Generated answer ({len(result.answer)} chars)")
        return result

    # ── Sync Convenience ─────────────────────────────────────────

    def query_sync(self, question: str, **kwargs: Any) -> RagResult:
        """Synchronous query convenience method.

        Runs the async query in a new event loop. Prefer ``query()``
        for production use.
        """
        import asyncio

        return asyncio.run(self.query(question, **kwargs))

    def ingest_sync(self, source: str, **kwargs: Any) -> IngestionResult:
        """Synchronous ingest convenience method."""
        import asyncio

        return asyncio.run(self.ingest(source, **kwargs))
