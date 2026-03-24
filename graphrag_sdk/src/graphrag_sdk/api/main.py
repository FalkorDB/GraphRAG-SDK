# GraphRAG SDK 2.0 — API: GraphRAG Facade
# Pattern: Facade — single entry point that hides all internal wiring.
# Principle: Simplicity — two-line usage: init + query/ingest.

from __future__ import annotations

import logging
import re
from typing import Any

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphSchema,
    IngestionResult,
    RagResult,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
    GraphExtraction,
)
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader
from graphrag_sdk.ingestion.loaders.text_loader import TextLoader
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.ingestion.resolution_strategies.base import ResolutionStrategy
from graphrag_sdk.ingestion.resolution_strategies.exact_match import ExactMatchResolution
from graphrag_sdk.retrieval.reranking_strategies.base import RerankingStrategy
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


_RAG_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY "
    "the provided context.\n\n"
    "RULES:\n"
    "1. Base your answer strictly on the context below.\n"
    "2. Be direct and concise — match your answer length to the "
    "question's complexity. A simple factual question deserves a short "
    "answer; a complex question may need more detail.\n"
    "3. Do not quote source passages verbatim. Do not include source "
    "references or citations in your answer.\n"
    "4. Do not start with preambles like 'According to the context' or "
    "'Based on the passage'. Just answer directly.\n"
    "5. Preserve exact names, dates, places, and factual details "
    "from the context.\n"
    "6. If the context lacks sufficient information, say so briefly "
    "rather than inventing details.\n"
    "7. Respect negation: if a passage states something did NOT happen "
    "or is NOT true, preserve that meaning.\n\n"
    "{context}\n\n"
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
        retrieval_strategy: Default retrieval strategy (uses MultiPathRetrieval if None).

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
        embedding_dimension: int = 1536,
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
            embedding_dimension=embedding_dimension,
        )

        # Deduplication engine
        self._deduplicator = EntityDeduplicator(self.graph_store, self.embedder)

        # Default retrieval strategy
        self._retrieval_strategy = retrieval_strategy or MultiPathRetrieval(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm=self.llm,
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
        - Extractor: GraphExtraction with configured LLM
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
            extractor=extractor or self._default_extractor(),
            resolver=resolver or ExactMatchResolution(),
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            schema=self.schema,
        )

        result = await pipeline.run(source, ctx, text=text)

        # Post-ingestion: create indices only.
        # backfill_entity_embeddings() is intentionally NOT called here —
        # it re-scans all entities and is very slow when ingesting multiple
        # documents sequentially.  Call it explicitly after all ingestion
        # is complete (e.g. ``await rag.vector_store.backfill_entity_embeddings()``).
        await self.vector_store.ensure_indices()

        return result

    def _default_extractor(self) -> ExtractionStrategy:
        """Return default GraphExtraction with schema entity types if available."""
        entity_types = [e.label for e in self.schema.entities] if self.schema.entities else None
        return GraphExtraction(
            llm=self.llm,
            entity_types=entity_types,
        )

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
            answer=self._clean_answer(llm_response.content),
            retriever_result=retriever_result if return_context else None,
            metadata={
                "model": self.llm.model_name,
                "num_context_items": len(retriever_result.items),
                "strategy": retrieval.__class__.__name__,
            },
        )

        ctx.log(f"Generated answer ({len(result.answer)} chars)")
        return result

    # ── Answer Post-processing ─────────────────────────────────

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Strip preambles, citations, and formatting artifacts from answers."""
        # Remove leading "Answer:" if echoed
        text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE)
        # Remove common preambles
        text = re.sub(
            r"^(?:According to|Based on|From) (?:the )?(?:provided |given )?"
            r"(?:context|passages?|texts?|narratives?|information|sources?"
            r"|documents?|retrieved)[,.]?\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        # Remove inline source citations: (Source: ...) or [Source: ...]
        text = re.sub(r"\s*\(Source:?\s*[^)]*\)", "", text)
        text = re.sub(r"\s*\[Source:?\s*[^\]]*\]", "", text)
        return text.strip()

    # ── Post-ingestion Operations ────────────────────────────────

    async def deduplicate_entities(
        self,
        *,
        fuzzy: bool = False,
        similarity_threshold: float = 0.9,
        batch_size: int = 500,
    ) -> int:
        """Global entity deduplication across all ingested documents.

        Phase 1 (always): Exact name match — groups entities by normalized
        name (lowercase, stripped), keeps the one with the longest description,
        remaps all RELATES and MENTIONED_IN edges, deletes duplicates.

        Phase 2 (optional): Fuzzy embedding match — embeds entity names,
        finds near-duplicates by cosine similarity, merges those too.

        Call after all documents are ingested.

        Args:
            fuzzy: If True, also perform fuzzy embedding-based dedup.
            similarity_threshold: Cosine similarity threshold for fuzzy dedup.
            batch_size: Entities per query batch.

        Returns:
            Total number of duplicate entities merged.
        """
        return await self._deduplicator.deduplicate(
            fuzzy=fuzzy,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
        )

    async def finalize(self) -> dict[str, Any]:
        """Run all post-ingestion steps after all documents are ingested.

        Bundles:
        1. Remove NULL-name stub entities (legacy cleanup)
        2. ``deduplicate_entities()`` — global exact-name dedup
        3. ``backfill_entity_embeddings()`` — name-only embeddings
        4. ``embed_relationships()`` — fact text embeddings on RELATES edges
        5. ``ensure_indices()`` — all indexes

        Returns:
            Dict with counts from each step.
        """
        ctx_log = logger.info

        ctx_log("finalize: starting post-ingestion steps")

        # Step 1: Remove NULL-name stub entities (created by legacy path-MERGE bugs)
        r = await self.graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.name IS NULL DETACH DELETE e RETURN count(e)"
        )
        null_cleaned = r.result_set[0][0] if r.result_set else 0
        if null_cleaned:
            ctx_log(f"finalize: removed {null_cleaned} NULL-name stub entities")

        # Step 2: Global dedup
        dedup_count = await self.deduplicate_entities()
        ctx_log(f"finalize: deduplicated {dedup_count} entities")

        # Step 3: Entity embeddings (name-only)
        entity_count = await self.vector_store.backfill_entity_embeddings()
        ctx_log(f"finalize: embedded {entity_count} entities")

        # Step 4: Relationship embeddings (fact text on RELATES edges)
        rel_count = await self.vector_store.embed_relationships()
        ctx_log(f"finalize: embedded {rel_count} relationships")

        # Step 5: Ensure all indexes
        self.vector_store._indices_ensured = False  # force re-check
        index_results = await self.vector_store.ensure_indices()
        ctx_log(f"finalize: indexes = {index_results}")

        return {
            "null_stubs_removed": null_cleaned,
            "entities_deduplicated": dedup_count,
            "entities_embedded": entity_count,
            "relationships_embedded": rel_count,
            "indexes": index_results,
        }

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

    def finalize_sync(self) -> dict[str, Any]:
        """Synchronous finalize convenience method."""
        import asyncio

        return asyncio.run(self.finalize())
