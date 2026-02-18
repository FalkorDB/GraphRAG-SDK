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
    GraphRelationship,
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
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


_RAG_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY "
    "the provided context.\n\n"
    "RULES:\n"
    "1. Base your answer strictly on the context below. "
    "Quote or closely paraphrase the source passages when possible.\n"
    "2. Combine information from entities, relationships, facts, and "
    "source passages to build a complete answer.\n"
    "3. Be specific — include names, dates, places, and details "
    "found in the context.\n"
    "4. If the context contains relevant information but not a direct "
    "answer, synthesize what IS available rather than guessing.\n"
    "5. If the context truly lacks information to answer, say so briefly "
    "rather than inventing details.\n"
    "6. The SOURCE DOCUMENT PASSAGES are the original text — prefer them "
    "over extracted entities/facts for specific details.\n"
    "7. Respect negation: if a passage states something did NOT happen "
    "or is NOT true, preserve that meaning in your answer.\n"
    "8. Be concise — answer in 1-3 sentences unless the question "
    "explicitly asks for a detailed explanation.\n\n"
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

        # Default retrieval strategy (llm_rerank=False: cosine reranker is sufficient,
        # LLM reranker adds ~1s latency with no measurable accuracy gain)
        self._retrieval_strategy = retrieval_strategy or MultiPathRetrieval(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm=self.llm,
            llm_rerank=False,
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
            skip_synonymy=True,
        )

        result = await pipeline.run(source, ctx, text=text)

        # Post-ingestion: create indices only.
        # backfill_entity_embeddings() is intentionally NOT called here —
        # it re-scans all entities and is very slow when ingesting multiple
        # documents sequentially.  Call it explicitly after all ingestion
        # is complete (e.g. ``await rag.vector_store.backfill_entity_embeddings()``).
        await self.vector_store.ensure_indices()

        return result

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

    # ── Post-ingestion Operations ────────────────────────────────

    async def detect_synonymy(
        self,
        *,
        similarity_threshold: float = 0.9,
        batch_size: int = 500,
    ) -> int:
        """Detect synonym entities and create SYNONYM edges (post-ingestion).

        Queries all ``__Entity__`` nodes, embeds their names, computes
        pairwise cosine similarity (block-wise to limit memory), and
        writes SYNONYM edges for pairs exceeding the threshold.

        Call this once after all documents have been ingested.

        Args:
            similarity_threshold: Minimum cosine similarity for a SYNONYM edge.
            batch_size: Entities per embedding batch.

        Returns:
            Number of SYNONYM edges created.
        """
        import numpy as np

        # Fetch all entity nodes
        offset = 0
        all_ids: list[str] = []
        all_names: list[str] = []
        while True:
            result = await self.graph_store.query_raw(
                "MATCH (e:__Entity__) "
                "RETURN e.id AS id, e.name AS name "
                "SKIP $offset LIMIT $limit",
                {"offset": offset, "limit": batch_size},
            )
            if not result.result_set:
                break
            for row in result.result_set:
                all_ids.append(row[0])
                all_names.append(row[1] if len(row) > 1 and row[1] else str(row[0]))
            offset += batch_size

        if len(all_ids) < 2:
            return 0

        # Batch embed all entity names
        raw_vectors = await self.embedder.aembed_documents(all_names)

        # Filter out entities whose embedding failed (None)
        valid = [
            (eid, name, vec)
            for eid, name, vec in zip(all_ids, all_names, raw_vectors)
            if vec is not None
        ]
        if len(valid) < 2:
            return 0
        all_ids, all_names, vectors = zip(*valid)  # type: ignore[assignment]
        all_ids = list(all_ids)

        # Block-wise cosine similarity
        mat = np.array(vectors, dtype=np.float32)
        norms_arr = np.linalg.norm(mat, axis=1, keepdims=True)
        norms_arr[norms_arr == 0] = 1.0
        mat_normed = mat / norms_arr

        BLOCK_SIZE = 1000
        n = len(all_ids)
        synonym_rels: list[GraphRelationship] = []
        for i_start in range(0, n, BLOCK_SIZE):
            i_end = min(i_start + BLOCK_SIZE, n)
            block = mat_normed[i_start:i_end]
            remaining = mat_normed[i_start:]
            sim_block = block @ remaining.T
            local_rows, local_cols = np.where(sim_block >= similarity_threshold)
            for lr, lc in zip(local_rows.tolist(), local_cols.tolist()):
                gi = i_start + lr
                gj = i_start + lc
                if gj > gi:
                    synonym_rels.append(GraphRelationship(
                        start_node_id=all_ids[gi],
                        end_node_id=all_ids[gj],
                        type="SYNONYM",
                        properties={"similarity": float(mat_normed[gi] @ mat_normed[gj])},
                    ))

        if synonym_rels:
            await self.graph_store.upsert_relationships(synonym_rels)

        logger.info(f"detect_synonymy: {len(synonym_rels)} SYNONYM edges from {n} entities")
        return len(synonym_rels)

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
