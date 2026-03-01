# GraphRAG SDK 2.0 — Retrieval: Multi-Path Strategy (Lean)
# Lean retrieval with RELATES edge vector search for facts + entity
# entry points, 2-path entity discovery, 4-path chunk retrieval,
# and cosine reranking.

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

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

        # 2. Embed question only
        query_vector = await self._embedder.aembed_query(query)

        # 3. RELATES edge vector search
        fact_strings, rel_entities = await self._search_relates_edges(query_vector)

        # 4. Entity discovery (2 paths) + merge rel_entities
        found_entities, entity_sources = await self._discover_entities(
            llm_kw, all_keywords
        )
        for eid, einfo in rel_entities.items():
            if eid not in found_entities:
                found_entities[eid] = einfo
                entity_sources[eid] = "rel_vector"

        # 5. Relationship expansion
        entity_list = list(found_entities.items())[:self._max_entities]
        relationship_strings = await self._expand_relationships(entity_list)

        # 6. Chunk retrieval (4 paths)
        candidate_chunks, chunk_sources = await self._retrieve_chunks(
            query, query_vector, llm_kw, simple_kw, entity_list
        )

        # 7. Source document names
        chunk_doc_map = await self._fetch_chunk_documents(list(candidate_chunks.keys()))

        # 8. Cosine rerank
        source_passages = await self._rerank_chunks(query_vector, candidate_chunks)

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

        # 9. Detect question type + assemble
        q_type_hint = self._detect_question_type(query)
        return self._assemble_raw_result(
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

    # -- Private helpers --

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

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

    async def _search_relates_edges(
        self, query_vector: list[float]
    ) -> tuple[list[str], dict[str, dict]]:
        """Search RELATES edges by vector similarity.

        Returns:
            fact_strings: ["src -[type]-> tgt: fact_text", ...]
            entities: dict of entity_id -> {name, description} discovered
                      from matched edge endpoints.
        """
        fact_strings: list[str] = []
        entities: dict[str, dict] = {}
        try:
            results = await self._vector.search_relationships(
                query_vector, top_k=self._rel_top_k
            )
            for rel in results:
                src = rel.get("src_name", "")
                tgt = rel.get("tgt_name", "")
                rel_type = rel.get("type", "")
                fact = rel.get("fact", "")
                if src and rel_type and tgt:
                    line = f"{src} —[{rel_type}]→ {tgt}"
                    if fact:
                        line += f": {fact}"
                    fact_strings.append(line)
                # Add entities as graph entry points
                if src:
                    src_id = src.strip().lower().replace(" ", "_")
                    if src_id not in entities:
                        entities[src_id] = {"name": src, "description": ""}
                if tgt:
                    tgt_id = tgt.strip().lower().replace(" ", "_")
                    if tgt_id not in entities:
                        entities[tgt_id] = {"name": tgt, "description": ""}
        except Exception:
            pass
        return fact_strings, entities

    async def _discover_entities(
        self,
        llm_kw: list[str],
        all_keywords: list[str],
    ) -> tuple[dict[str, dict], dict[str, str]]:
        """2-path entity discovery.

        Paths:
        a: Cypher CONTAINS on entity names
        b: Fulltext search on entity index
        """
        found: dict[str, dict] = {}
        sources: dict[str, str] = {}

        def _add(eid: str, info: dict, source: str) -> None:
            if eid and eid not in found:
                found[eid] = info
                sources[eid] = source

        # Path a: Cypher CONTAINS on entity names (batched UNWIND)
        kw_batch = [kw for kw in llm_kw[:8] if kw]
        if kw_batch:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $keywords AS kw "
                    "MATCH (e:__Entity__) WHERE toLower(e.name) CONTAINS toLower(kw) "
                    "RETURN e.id AS id, e.name AS name, e.description AS desc "
                    "LIMIT 40",
                    {"keywords": kw_batch},
                )
                for row in result.result_set:
                    _add(
                        row[0],
                        {"name": row[1] if len(row) > 1 else "", "description": row[2] if len(row) > 2 else ""},
                        "cypher_contains",
                    )
            except Exception:
                pass

        # Path b: Fulltext search on entity index
        for kw in all_keywords[:6]:
            try:
                ft_ents = await self._vector.fulltext_search(kw, top_k=3, label="__Entity__")
                for ent in ft_ents:
                    eid = ent.get("id", "")
                    if eid:
                        try:
                            detail = await self._graph.query_raw(
                                "MATCH (e:__Entity__ {id: $eid}) "
                                "RETURN e.name AS name, e.description AS desc",
                                {"eid": eid},
                            )
                            if detail.result_set:
                                row = detail.result_set[0]
                                _add(eid, {
                                    "name": row[0] if row[0] else "",
                                    "description": row[1] if len(row) > 1 and row[1] else "",
                                }, "fulltext")
                        except Exception:
                            _add(eid, {"name": "", "description": ""}, "fulltext")
            except Exception:
                pass

        return found, sources

    async def _expand_relationships(
        self, entity_list: list[tuple[str, dict]]
    ) -> list[str]:
        """1-hop + 2-hop relationship expansion from top entities.

        Uses the single ``RELATES`` edge type — no need to filter out
        SYNONYM or MENTIONED_IN edges. The ``rel_type`` property stores
        the original relationship type, and ``fact`` stores the evidence.
        """
        relationship_strings: list[str] = []
        seen: set[tuple] = set()

        # 1-hop relationships (batched UNWIND)
        eids_1hop = [eid for eid, _ in entity_list[:15]]
        if eids_1hop:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (a:__Entity__ {id: eid})-[r:RELATES]->(b:__Entity__) "
                    "RETURN a.name AS src, r.rel_type AS rel, b.name AS tgt, "
                    "COALESCE(r.fact, r.description, '') AS fact "
                    "LIMIT 150",
                    {"eids": eids_1hop},
                )
                for row in result.result_set:
                    src = row[0] or ""
                    rel_type = row[1] if len(row) > 1 else ""
                    tgt = row[2] if len(row) > 2 else ""
                    fact = row[3] if len(row) > 3 else ""
                    key = (src.lower(), rel_type, tgt.lower())
                    if src and rel_type and tgt and key not in seen:
                        seen.add(key)
                        line = f"{src} —[{rel_type}]→ {tgt}"
                        if fact:
                            line += f": {fact}"
                        relationship_strings.append(line)
            except Exception:
                pass

        # 2-hop relationships for top 5 entities (batched UNWIND)
        eids_2hop = [eid for eid, _ in entity_list[:5]]
        if eids_2hop:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (a:__Entity__ {id: eid})-[r1:RELATES]->(b:__Entity__)"
                    "-[r2:RELATES]->(c:__Entity__) "
                    "RETURN a.name, r1.rel_type, b.name, r2.rel_type, c.name "
                    "LIMIT 25",
                    {"eids": eids_2hop},
                )
                for row in result.result_set:
                    a_name = row[0] or ""
                    r1_type = row[1] if len(row) > 1 else ""
                    b_name = row[2] if len(row) > 2 else ""
                    r2_type = row[3] if len(row) > 3 else ""
                    c_name = row[4] if len(row) > 4 else ""
                    if a_name and r1_type and b_name and r2_type and c_name:
                        key = (a_name.lower(), r1_type, b_name.lower(), r2_type, c_name.lower())
                        if key not in seen:
                            seen.add(key)
                            line = f"{a_name} —[{r1_type}]→ {b_name} —[{r2_type}]→ {c_name}"
                            relationship_strings.append(line)
            except Exception:
                pass

        return relationship_strings[:self._max_relationships]

    async def _retrieve_chunks(
        self,
        query: str,
        query_vector: list[float],
        llm_kw: list[str],
        simple_kw: list[str],
        entity_list: list[tuple[str, dict]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """4-path chunk retrieval: fulltext + vector + MENTIONED_IN + 2-hop."""
        chunks: dict[str, str] = {}
        sources: dict[str, str] = {}

        def _add(cid: str, text: str, source: str) -> None:
            if cid and text and cid not in chunks:
                chunks[cid] = text
                sources[cid] = source

        # Path A: Fulltext search
        fulltext_queries = [query] + llm_kw[:6] + simple_kw[:4]
        for ft_q in fulltext_queries:
            try:
                results = await self._vector.fulltext_search(ft_q, top_k=5, label="Chunk")
                for c in results:
                    _add(c.get("id", ""), c.get("text", ""), "fulltext")
            except Exception:
                pass

        # Path B: Vector search
        try:
            results = await self._vector.search(query_vector, top_k=15, label="Chunk")
            for c in results:
                _add(c.get("id", ""), c.get("text", ""), "vector")
        except Exception:
            pass

        # Path C: MENTIONED_IN — 3 chunks per entity (batched UNWIND)
        eids_mention = [eid for eid, _ in entity_list[:15]]
        if eids_mention:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (e:__Entity__ {id: eid})-[:MENTIONED_IN]->(c:Chunk) "
                    "WITH eid, COLLECT(c)[..3] AS chunks "
                    "UNWIND chunks AS c "
                    "RETURN eid, c.id AS id, c.text AS text",
                    {"eids": eids_mention},
                )
                for row in result.result_set:
                    cid = row[1]
                    text = row[2] if len(row) > 2 else ""
                    _add(cid, text, "mentioned_in")
            except Exception:
                pass

        # Path D: 2-hop entity→neighbor→chunk (batched UNWIND)
        eids_2hop_chunk = [eid for eid, _ in entity_list[:10]]
        if eids_2hop_chunk:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (e:__Entity__ {id: eid})-[:RELATES]-(neighbor:__Entity__)"
                    "-[:MENTIONED_IN]->(c:Chunk) "
                    "RETURN DISTINCT c.id AS id, c.text AS text "
                    "LIMIT 20",
                    {"eids": eids_2hop_chunk},
                )
                for row in result.result_set:
                    cid = row[0]
                    text = row[1] if len(row) > 1 else ""
                    _add(cid, text, "2hop_mentioned")
            except Exception:
                pass

        return chunks, sources

    async def _fetch_chunk_documents(
        self, chunk_ids: list[str]
    ) -> dict[str, str]:
        """Batch-fetch source document name for each chunk via PART_OF."""
        if not chunk_ids:
            return {}
        try:
            result = await self._graph.query_raw(
                "UNWIND $ids AS cid "
                "MATCH (d:Document)-[:PART_OF]->(c:Chunk {id: cid}) "
                "RETURN c.id AS cid, d.path AS path",
                {"ids": chunk_ids},
            )
            mapping: dict[str, str] = {}
            for row in result.result_set:
                cid = row[0] or ""
                path = row[1] if len(row) > 1 else ""
                if cid and path:
                    name = path.rsplit("/", 1)[-1] if "/" in path else path
                    mapping[cid] = name
            return mapping
        except Exception:
            return {}

    async def _rerank_chunks(
        self,
        query_vector: list[float],
        candidate_chunks: dict[str, str],
    ) -> list[str]:
        """Batch embed candidates and cosine-sort, take top_k."""
        if not candidate_chunks:
            return []

        chunk_texts = list(candidate_chunks.values())
        try:
            chunk_vectors = await self._embedder.aembed_documents(chunk_texts)
            scored = [
                (i, self._cosine_sim(query_vector, cvec))
                for i, cvec in enumerate(chunk_vectors)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [chunk_texts[i] for i, _ in scored[:self._chunk_top_k]]
        except Exception:
            return chunk_texts[:self._chunk_top_k]

    @staticmethod
    def _detect_question_type(query: str) -> str:
        """Detect question type and return an answer-format hint."""
        q = query.strip().lower()
        if q.startswith(("is ", "are ", "was ", "were ", "did ", "does ",
                         "do ", "has ", "had ", "have ", "can ", "could ",
                         "will ", "would ", "should ")):
            return "Answer format: This is a yes/no question — start with Yes or No, then explain briefly."
        if q.startswith("who "):
            return "Answer format: Name the specific person(s) or character(s)."
        if q.startswith("where "):
            return "Answer format: Name the specific place or location."
        if q.startswith("when "):
            return "Answer format: Provide the specific time, date, or period."
        if q.startswith("how many") or q.startswith("how much"):
            return "Answer format: Provide a specific number or quantity."
        return ""

    def _assemble_raw_result(
        self,
        entity_list: list[tuple[str, dict]],
        relationship_strings: list[str],
        fact_strings: list[str],
        source_passages: list[str],
        q_type_hint: str = "",
    ) -> RawSearchResult:
        """Build structured RawSearchResult with section records."""
        records: list[dict[str, Any]] = []

        # Question-type hint (prepended so LLM sees it first)
        if q_type_hint:
            records.append({
                "section": "hint",
                "content": q_type_hint,
            })

        # Entity section
        seen_names: set[str] = set()
        entity_lines: list[str] = []
        for _, einfo in entity_list:
            name = einfo.get("name", "")
            if name and name.lower() not in seen_names:
                seen_names.add(name.lower())
                desc = einfo.get("description", "")
                entity_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        if entity_lines:
            records.append({
                "section": "entities",
                "content": "## Key Entities\n" + "\n".join(entity_lines[:25]),
            })

        # Relationship section
        if relationship_strings:
            records.append({
                "section": "relationships",
                "content": "## Entity Relationships\n"
                + "\n".join(f"- {r}" for r in relationship_strings[:20]),
            })

        # Knowledge Graph Facts section (from RELATES edge vector search)
        if fact_strings:
            records.append({
                "section": "facts",
                "content": "## Knowledge Graph Facts\n"
                + "\n".join(f"- {f}" for f in fact_strings[:15]),
            })

        # Passages section
        if source_passages:
            records.append({
                "section": "passages",
                "content": "## Source Document Passages\n"
                + "\n---\n".join(source_passages[:15]),
            })

        return RawSearchResult(
            records=records,
            metadata={"strategy": "multi_path"},
        )
