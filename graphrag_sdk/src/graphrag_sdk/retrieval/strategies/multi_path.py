# GraphRAG SDK 2.0 — Retrieval: Multi-Path Strategy (Lean)
# Thin orchestrator combining RELATES edge vector search for facts +
# entity entry points, 2-path entity discovery, 4-path chunk retrieval,
# and cosine reranking.  Phase logic lives in sub-modules.

from __future__ import annotations

import logging
import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.chunk_retrieval import (
    fetch_chunk_documents,
    retrieve_chunks,
)
from graphrag_sdk.retrieval.strategies.entity_discovery import (
    discover_entities,
    search_relates_edges,
)
from graphrag_sdk.retrieval.strategies.relationship_expansion import (
    expand_relationships,
)
from graphrag_sdk.retrieval.strategies.result_assembly import (
    assemble_raw_result,
    cosine_sim,
    detect_question_type,
    rerank_chunks,
)

logger = logging.getLogger(__name__)


class MultiPathRetrieval(RetrievalStrategy):
    """Multi-path retrieval combining RELATES edge vector search, fulltext,
    graph traversal, and cosine reranking.

    Retrieval pipeline:
      1. Keyword extraction (stopword filter + LLM proper nouns)
      2. Embed question only (single API call)
      3. RELATES edge vector search -> fact strings + entity entry points
      4. Entity discovery (2 paths: Cypher CONTAINS, fulltext)
         + merge entities from step 3
      5. Relationship expansion (1-hop + 2-hop from top entities)
      6. Chunk retrieval (4 paths: fulltext, vector, MENTIONED_IN, 2-hop)
      7. Fetch source document names
      8. Cosine reranking of all candidate chunks
      9. Context assembly into structured sections (hint, entities,
         relationships, facts, passages)

    Args:
        graph_store: Graph data access object.
        vector_store: Vector data access object.
        embedder: Embedding provider.
        llm: LLM provider for keyword extraction.
        chunk_top_k: Final chunks after reranking (default: 15).
        max_entities: Max entities to keep (default: 30).
        max_relationships: Max relationships to keep (default: 20).
        rel_top_k: RELATES edge vector search results (default: 15).
        keyword_limit: Max keywords to extract (default: 10).
    """

    _STOP_WORDS = frozenset({
        "what", "who", "where", "when", "why", "how", "which", "whom",
        "is", "are", "was", "were", "be", "been", "being",
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
        "or", "with", "by", "from", "as", "but", "not", "no", "nor",
        "does", "did", "do", "has", "had", "have", "will", "would",
        "could", "should", "may", "might", "shall", "can",
        "this", "that", "these", "those", "it", "its", "they", "their",
        "he", "she", "him", "her", "his", "about", "after", "before",
        "between", "during", "through", "according", "described",
    })

    def __init__(
        self,
        graph_store: Any,
        vector_store: Any,
        embedder: Embedder,
        llm: LLMInterface,
        *,
        chunk_top_k: int = 15,
        max_entities: int = 30,
        max_relationships: int = 20,
        rel_top_k: int = 15,
        keyword_limit: int = 10,
    ) -> None:
        super().__init__(graph_store=graph_store, vector_store=vector_store)
        self._embedder = embedder
        self._llm = llm
        self._chunk_top_k = chunk_top_k
        self._max_entities = max_entities
        self._max_relationships = max_relationships
        self._rel_top_k = rel_top_k
        self._keyword_limit = keyword_limit

    # -- Template Method hook --

    async def _execute(
        self,
        query: str,
        ctx: Context,
        **kwargs: Any,
    ) -> RawSearchResult:
        # 1. Extract keywords
        simple_kw, llm_kw = await self._extract_keywords(query)
        all_keywords = llm_kw[:8] + simple_kw
        ctx.log(f"MultiPath [1/9]: {len(all_keywords)} keywords extracted")

        # 2. Embed question only
        query_vector = await self._embedder.aembed_query(query)

        # 3. RELATES edge vector search
        fact_strings, rel_entities = await search_relates_edges(
            self._vector, query_vector, self._rel_top_k
        )
        ctx.log(f"MultiPath [3/9]: {len(fact_strings)} facts, {len(rel_entities)} rel-entities")

        # 4. Entity discovery (2 paths) + merge rel_entities
        found_entities, entity_sources = await discover_entities(
            self._graph, self._vector, llm_kw, all_keywords
        )
        for eid, einfo in rel_entities.items():
            if eid not in found_entities:
                found_entities[eid] = einfo
                entity_sources[eid] = "rel_vector"
        ctx.log(f"MultiPath [4/9]: {len(found_entities)} entities discovered")

        # 5. Relationship expansion
        entity_list = list(found_entities.items())[:self._max_entities]
        relationship_strings = await expand_relationships(
            self._graph, entity_list, self._max_relationships
        )
        ctx.log(f"MultiPath [5/9]: {len(relationship_strings)} relationships")

        # 6. Chunk retrieval (4 paths)
        candidate_chunks, chunk_sources = await retrieve_chunks(
            self._vector, self._graph, query, query_vector,
            llm_kw, simple_kw, entity_list,
        )
        ctx.log(f"MultiPath [6/9]: {len(candidate_chunks)} candidate chunks")

        # 7. Source document names
        chunk_doc_map = await fetch_chunk_documents(
            self._graph, list(candidate_chunks.keys())
        )

        # 8. Cosine rerank
        source_passages = await rerank_chunks(
            self._embedder, query_vector, candidate_chunks, self._chunk_top_k
        )

        # Tag with source docs
        text_to_doc: dict[str, str] = {
            candidate_chunks[cid]: doc_name
            for cid, doc_name in chunk_doc_map.items()
            if cid in candidate_chunks
        }
        source_passages = [
            f"[Source: {text_to_doc[p]}]\n{p}" if p in text_to_doc else p
            for p in source_passages
        ]
        ctx.log(f"MultiPath [8/9]: {len(source_passages)} passages after rerank")

        # 9. Detect question type + assemble
        q_type_hint = detect_question_type(query)
        return assemble_raw_result(
            entity_list, relationship_strings, fact_strings,
            source_passages, q_type_hint,
        )

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        """Produce RetrieverResultItems as markdown sections."""
        items: list[RetrieverResultItem] = []
        for record in raw.records:
            section_type = record.get("section", "")
            content = record.get("content", "")
            if content:
                items.append(
                    RetrieverResultItem(
                        content=content,
                        metadata={"section": section_type},
                    )
                )
        return RetrieverResult(items=items, metadata=raw.metadata)

    # -- Internal: keyword extraction (stays in orchestrator) --

    async def _extract_keywords(
        self, query: str
    ) -> tuple[list[str], list[str]]:
        """Extract simple + LLM-based keywords from the query."""
        words = re.sub(r"[?.!,;:'\"-]", " ", query.lower()).split()
        simple = [w for w in words if w not in self._STOP_WORDS and len(w) > 2][:12]

        llm_kw: list[str] = []
        try:
            response = await self._llm.ainvoke(
                "Extract ALL proper nouns, character names, person names, place names, "
                "book titles, and specific terms from this question. "
                "Return them comma-separated, nothing else.\n\n"
                f"Question: {query}\n\nNames: "
            )
            llm_kw = [
                k.strip().strip("'\"")
                for k in response.content.split(",")
                if k.strip() and len(k.strip()) > 1
            ]
        except Exception as exc:
            logger.debug(f"LLM keyword extraction failed: {exc}")

        return simple, llm_kw

    # -- Backward-compatibility wrappers (used by tests) --

    async def _search_relates_edges(
        self, query_vector: list[float]
    ) -> tuple[list[str], dict[str, dict]]:
        """Backward-compat wrapper — delegates to module function."""
        return await search_relates_edges(
            self._vector, query_vector, self._rel_top_k
        )

    @staticmethod
    def _detect_question_type(query: str) -> str:
        """Backward-compat wrapper — delegates to module function."""
        return detect_question_type(query)

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Backward-compat wrapper — delegates to module function."""
        return cosine_sim(a, b)
