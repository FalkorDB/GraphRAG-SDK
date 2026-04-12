# GraphRAG SDK — API: GraphRAG Facade
# Pattern: Facade — single entry point that hides all internal wiring.
# Principle: Simplicity — two-line usage: init + query/ingest.

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, overload

from graphrag_sdk import __version__
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import ConfigError
from graphrag_sdk.core.models import (
    ChatMessage,
    GraphSchema,
    IngestionResult,
    RagResult,
    RetrieverResult,
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
    "or is NOT true, preserve that meaning.\n"
    "{history_section}\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class GraphRAG:
    """The main user-facing class for GraphRAG operations.

    Provides three primary operations:
    - ``ingest()`` — build a knowledge graph from sources
    - ``retrieve()`` — search the knowledge graph (retrieval only)
    - ``completion()`` — retrieve context and generate an answer

    All use sensible defaults but allow full strategy customisation.

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

        # Retrieve context only
        context = await rag.retrieve("What is the capital of France?")

        # Full RAG: retrieve + generate answer
        result = await rag.completion("What is the capital of France?")
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
        self._embedding_dimension = embedding_dimension
        self._config_validated = False

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

    @overload
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
    ) -> IngestionResult: ...

    @overload
    async def ingest(
        self,
        source: list[str],
        *,
        loader: LoaderStrategy | None = None,
        chunker: ChunkingStrategy | None = None,
        extractor: ExtractionStrategy | None = None,
        resolver: ResolutionStrategy | None = None,
        max_concurrent: int = 3,
        ctx: Context | None = None,
    ) -> list[IngestionResult]: ...

    async def ingest(
        self,
        source: str | list[str],
        *,
        text: str | None = None,
        loader: LoaderStrategy | None = None,
        chunker: ChunkingStrategy | None = None,
        extractor: ExtractionStrategy | None = None,
        resolver: ResolutionStrategy | None = None,
        max_concurrent: int = 3,
        ctx: Context | None = None,
    ) -> IngestionResult | list[IngestionResult]:
        """Build a knowledge graph from one or more sources.

        Accepts a single source path or a list of paths. When a list
        is provided, documents are ingested in parallel with bounded
        concurrency.

        Uses sensible defaults for any unspecified strategy:
        - Loader: auto-detected from file extension (PDF or text)
        - Chunker: FixedSizeChunking(chunk_size=1000)
        - Extractor: GraphExtraction with configured LLM
        - Resolver: ExactMatchResolution

        Args:
            source: File path (or list of paths) for ingestion.
            text: Optional raw text (single source only).
            loader: Custom loader strategy.
            chunker: Custom chunking strategy.
            extractor: Custom extraction strategy.
            resolver: Custom resolution strategy.
            max_concurrent: Max parallel ingestions (default 3).
            ctx: Execution context.

        Returns:
            IngestionResult for a single source, or
            list[IngestionResult] for multiple sources.
        """
        if isinstance(source, list):
            if text is not None:
                raise ValueError("'text' parameter cannot be used with a list of sources")
            return await self._ingest_batch(
                source,
                loader=loader,
                chunker=chunker,
                extractor=extractor,
                resolver=resolver,
                max_concurrent=max_concurrent,
                ctx=ctx,
            )
        return await self._ingest_single(
            source,
            text=text,
            loader=loader,
            chunker=chunker,
            extractor=extractor,
            resolver=resolver,
            ctx=ctx,
        )

    async def _ingest_single(
        self,
        source: str,
        *,
        text: str | None = None,
        loader: LoaderStrategy | None = None,
        chunker: ChunkingStrategy | None = None,
        extractor: ExtractionStrategy | None = None,
        resolver: ResolutionStrategy | None = None,
        ctx: Context | None = None,
        _skip_post: bool = False,
    ) -> IngestionResult:
        """Ingest a single source document.

        Args:
            _skip_post: Internal flag — when True, skips ensure_indices
                and config write (caller handles them after the batch).
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

        if not _skip_post:
            # Post-ingestion: create indices only.
            # backfill_entity_embeddings() is intentionally NOT called here —
            # it re-scans all entities and is very slow when ingesting multiple
            # documents sequentially.  Call finalize() after all ingestion.
            await self.vector_store.ensure_indices()

            # Write/update graph config node (idempotent)
            await self._write_graph_config()

        return result

    async def _ingest_batch(
        self,
        sources: list[str],
        *,
        loader: LoaderStrategy | None = None,
        chunker: ChunkingStrategy | None = None,
        extractor: ExtractionStrategy | None = None,
        resolver: ResolutionStrategy | None = None,
        max_concurrent: int = 3,
        ctx: Context | None = None,
    ) -> list[IngestionResult]:
        """Ingest multiple sources in parallel with bounded concurrency."""
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

        parent_ctx = ctx or Context()
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(src: str) -> IngestionResult:
            async with sem:
                return await self._ingest_single(
                    src,
                    text=None,
                    loader=loader,
                    chunker=chunker,
                    extractor=extractor,
                    resolver=resolver,
                    ctx=parent_ctx.child(),
                    _skip_post=True,
                )

        results = list(await asyncio.gather(*[_one(s) for s in sources]))

        # Post-batch: run ensure_indices and config write once
        await self.vector_store.ensure_indices()
        await self._write_graph_config()

        return results

    def _default_extractor(self) -> ExtractionStrategy:
        """Return default GraphExtraction with schema entity types if available."""
        entity_types = [e.label for e in self.schema.entities] if self.schema.entities else None
        return GraphExtraction(
            llm=self.llm,
            entity_types=entity_types,
        )

    # ── Retrieval ────────────────────────────────────────────────

    async def retrieve(
        self,
        question: str,
        *,
        strategy: RetrievalStrategy | None = None,
        reranker: RerankingStrategy | None = None,
        ctx: Context | None = None,
    ) -> RetrieverResult:
        """Retrieve context from the knowledge graph without generating an answer.

        Use this to inspect retrieved context or pass it to your own LLM.
        Use ``completion()`` for the full RAG pipeline.

        Args:
            question: The user's question.
            strategy: Override retrieval strategy (uses default if None).
            reranker: Optional reranking strategy to apply.
            ctx: Execution context.

        Returns:
            RetrieverResult with context items and metadata.
        """
        if ctx is None:
            ctx = Context()

        ctx.log(f"Retrieve: {question[:80]}...")

        await self._validate_graph_config()

        retrieval = strategy or self._retrieval_strategy
        retriever_result = await retrieval.search(question, ctx)

        if reranker is not None:
            retriever_result = await reranker.rerank(question, retriever_result, ctx)

        ctx.log(f"Retrieved {len(retriever_result.items)} context items")
        return retriever_result

    # ── Completion ──────────────────────────────────────────────

    @staticmethod
    def _validate_history(
        history: list[ChatMessage | dict[str, str]],
    ) -> list[ChatMessage]:
        """Validate and normalise conversation history to ``ChatMessage`` objects.

        Accepts either pre-built ``ChatMessage`` instances or plain dicts
        with ``role`` and ``content`` keys.  Raises ``ValueError`` on
        invalid entries.
        """
        validated: list[ChatMessage] = []
        for i, msg in enumerate(history):
            if isinstance(msg, ChatMessage):
                validated.append(msg)
            elif isinstance(msg, dict):
                if "role" not in msg or "content" not in msg:
                    raise ValueError(
                        f"history[{i}]: each message must have 'role' and "
                        f"'content' keys, got {sorted(msg.keys())}"
                    )
                try:
                    validated.append(ChatMessage(role=msg["role"], content=msg["content"]))
                except Exception:
                    raise ValueError(
                        f"history[{i}]: invalid role '{msg['role']}'. "
                        f"Must be one of: 'system', 'user', 'assistant'"
                    )
            else:
                raise TypeError(
                    f"history[{i}]: expected ChatMessage or dict, got {type(msg).__name__}"
                )
        return validated

    async def completion(
        self,
        question: str,
        *,
        history: list[ChatMessage | dict[str, str]] | None = None,
        strategy: RetrievalStrategy | None = None,
        reranker: RerankingStrategy | None = None,
        prompt_template: str | None = None,
        return_context: bool = False,
        ctx: Context | None = None,
    ) -> RagResult:
        """Full RAG pipeline: retrieve context and generate an answer.

        Args:
            question: The user's question.
            history: Optional conversation history as a list of
                ``ChatMessage`` objects or ``{"role": ..., "content": ...}``
                dicts.  Supported roles: ``"system"``, ``"user"``,
                ``"assistant"``.  Messages are passed natively to the LLM
                provider's multi-turn API.
            strategy: Override retrieval strategy (uses default if None).
            reranker: Optional reranking strategy to apply.
            prompt_template: Custom prompt (must contain {context} and
                {question}).  Only used when *history* is ``None`` (single-turn).
            return_context: If True, include retriever results in output.
            ctx: Execution context.

        Returns:
            RagResult with the generated answer.
        """
        if ctx is None:
            ctx = Context()

        ctx.log(f"Completion: {question[:80]}...")

        # Step 1-2: Retrieve + rerank
        retriever_result = await self.retrieve(
            question,
            strategy=strategy,
            reranker=reranker,
            ctx=ctx,
        )

        # Step 3: Build context string
        context_str = "\n---\n".join(item.content for item in retriever_result.items)

        # Step 4: Generate answer
        if history:
            # Multi-turn: validate history and use native messages API
            validated = self._validate_history(history)
            system_prompt = (prompt_template or _RAG_PROMPT).format(
                context=context_str,
                question="",
                history_section="",
            )
            messages: list[ChatMessage] = [
                ChatMessage(role="system", content=system_prompt),
                *validated,
                ChatMessage(role="user", content=question),
            ]
            llm_response = await self.llm.ainvoke_messages(messages)
        else:
            # Single-turn: use the existing prompt template path
            template = prompt_template or _RAG_PROMPT
            prompt = template.format(
                context=context_str,
                question=question,
                history_section="",
            )
            llm_response = await self.llm.ainvoke(prompt)

        result = RagResult(
            answer=self._clean_answer(llm_response.content),
            retriever_result=retriever_result if return_context else None,
            metadata={
                "model": self.llm.model_name,
                "num_context_items": len(retriever_result.items),
                "strategy": (strategy or self._retrieval_strategy).__class__.__name__,
                "has_history": bool(history),
            },
        )

        ctx.log(f"Generated answer ({len(result.answer)} chars)")
        return result

    # ── Deprecated query() alias ────────────────────────────────

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
        """Deprecated: use ``completion()`` instead.

        This is a backward-compatibility alias. It will be removed in
        a future version.
        """
        import warnings

        warnings.warn(
            "GraphRAG.query() is deprecated. Use completion() for the full "
            "RAG pipeline, or retrieve() for retrieval-only.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.completion(
            question,
            strategy=strategy,
            reranker=reranker,
            prompt_template=prompt_template,
            return_context=return_context,
            ctx=ctx,
        )

    # ── Graph Config ────────────────────────────────────────────

    async def _write_graph_config(self) -> None:
        """Write or update the ``__GraphRAGConfig__`` singleton node."""
        from datetime import datetime, timezone

        try:
            await self.graph_store.query_raw(
                "MERGE (c:__GraphRAGConfig__ {id: 'default'}) "
                "SET c.embedding_model = $model, "
                "c.embedding_dimension = $dim, "
                "c.sdk_version = $version, "
                "c.updated_at = $ts",
                params={
                    "model": self.embedder.model_name,
                    "dim": self._embedding_dimension,
                    "version": __version__,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            logger.debug("Failed to write graph config node", exc_info=True)

    async def _validate_graph_config(self) -> None:
        """Check that the current embedder matches the graph's stored config."""
        if self._config_validated:
            return

        try:
            result = await self.graph_store.query_raw(
                "MATCH (c:__GraphRAGConfig__ {id: 'default'}) "
                "RETURN c.embedding_model, c.embedding_dimension"
            )
            if not result.result_set:
                # No config node — graph is empty or pre-config
                self._config_validated = True
                return

            stored_model = result.result_set[0][0]
            stored_dim = result.result_set[0][1]
            current_model = self.embedder.model_name

            if stored_model and stored_model != current_model:
                raise ConfigError(
                    f"Embedding model mismatch: graph was built with "
                    f"'{stored_model}' but current embedder is "
                    f"'{current_model}'. Use the same embedding model "
                    f"to query this graph."
                )
            if stored_dim and stored_dim != self._embedding_dimension:
                raise ConfigError(
                    f"Embedding dimension mismatch: graph was built with "
                    f"dimension {stored_dim} but current config is "
                    f"{self._embedding_dimension}."
                )
        except ConfigError:
            raise
        except Exception:
            # Don't mark as validated on transient failures — retry next call.
            logger.debug("Failed to validate graph config", exc_info=True)
            return

        self._config_validated = True

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

    def retrieve_sync(self, question: str, **kwargs: Any) -> RetrieverResult:
        """Synchronous retrieve convenience method."""
        return asyncio.run(self.retrieve(question, **kwargs))

    def completion_sync(self, question: str, **kwargs: Any) -> RagResult:
        """Synchronous completion convenience method."""
        return asyncio.run(self.completion(question, **kwargs))

    def query_sync(self, question: str, **kwargs: Any) -> RagResult:
        """Deprecated: use ``completion_sync()`` instead."""
        import warnings

        warnings.warn(
            "GraphRAG.query_sync() is deprecated. Use completion_sync().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.completion_sync(question, **kwargs)

    def ingest_sync(
        self, source: str | list[str], **kwargs: Any
    ) -> IngestionResult | list[IngestionResult]:
        """Synchronous ingest convenience method."""
        return asyncio.run(self.ingest(source, **kwargs))

    def finalize_sync(self) -> dict[str, Any]:
        """Synchronous finalize convenience method."""
        return asyncio.run(self.finalize())
