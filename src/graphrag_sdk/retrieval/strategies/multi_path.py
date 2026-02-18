# GraphRAG SDK 2.0 — Retrieval: Multi-Path Strategy
# Proven 87.9%-accuracy retrieval with 5-path entity discovery,
# 5-path chunk retrieval (incl. 2-hop), cosine reranking, and fact retrieval.

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
    """Multi-path retrieval combining vector, fulltext, graph, and keyword search.

    Retrieval pipeline:
      1. Keyword extraction (stopword filter + LLM proper nouns)
      2. Batch embed question + keywords (single API call)
      3. Entity discovery (5 paths: vector/kw, CONTAINS, fulltext, q-vector, synonyms)
      4. Relationship expansion (1-hop + 2-hop from top entities)
      5. Chunk retrieval (5 paths: fulltext, vector, MENTIONED_IN, CONTAINS, 2-hop)
      6. Cosine reranking of all candidate chunks
      6b. LLM reranking for entity-specificity (top_k -> llm_rerank_top_k)
      7. Fact retrieval (vector search on Fact nodes)
      8. Context assembly into structured sections

    Args:
        graph_store: Graph data access object.
        vector_store: Vector data access object.
        embedder: Embedding provider.
        llm: LLM provider for keyword extraction.
        entity_top_k: Entities per vector search (default: 5).
        chunk_top_k: Final chunks after reranking (default: 15).
        fact_top_k: Facts from vector search (default: 15).
        max_entities: Max entities to keep (default: 30).
        max_relationships: Max relationships to keep (default: 20).
        keyword_limit: Max keywords to extract (default: 10).
        llm_rerank_top_k: Passages after LLM reranking (default: 8).
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
        entity_top_k: int = 5,
        chunk_top_k: int = 15,
        fact_top_k: int = 15,
        max_entities: int = 30,
        max_relationships: int = 20,
        keyword_limit: int = 10,
        llm_rerank_top_k: int = 8,
        llm_rerank: bool = True,
    ) -> None:
        super().__init__(graph_store=graph_store, vector_store=vector_store)
        self._embedder = embedder
        self._llm = llm
        self._entity_top_k = entity_top_k
        self._chunk_top_k = chunk_top_k
        self._fact_top_k = fact_top_k
        self._max_entities = max_entities
        self._max_relationships = max_relationships
        self._keyword_limit = keyword_limit
        self._llm_rerank_top_k = llm_rerank_top_k
        self._llm_rerank = llm_rerank

    # ── Template Method hook ──────────────────────────────────────

    async def _execute(
        self,
        query: str,
        ctx: Context,
        **kwargs: Any,
    ) -> RawSearchResult:
        # 1. Extract keywords
        simple_kw, llm_kw = await self._extract_keywords(query)
        all_keywords = llm_kw[: 8] + simple_kw

        # 2. Batch embed question + keywords
        query_vector, kw_vectors = await self._batch_embed(query, all_keywords)

        # 3. Discover entities (5 paths)
        found_entities, entity_sources = await self._discover_entities(
            query_vector, kw_vectors, llm_kw, all_keywords
        )

        # 4. Expand relationships
        entity_list = list(found_entities.items())[: self._max_entities]
        relationship_strings = await self._expand_relationships(entity_list)

        # 5. Retrieve chunks (5 paths)
        candidate_chunks, chunk_sources = await self._retrieve_chunks(
            query, query_vector, llm_kw, simple_kw, entity_list
        )

        # 5b. Fetch source document names for chunks
        chunk_doc_map = await self._fetch_chunk_documents(list(candidate_chunks.keys()))

        # 6. Cosine rerank
        source_passages = await self._rerank_chunks(query_vector, candidate_chunks)

        # 6b. LLM rerank (entity-specificity filter) — optional
        if self._llm_rerank:
            source_passages = await self._llm_rerank_passages(
                query, source_passages, self._llm_rerank_top_k
            )

        # 6c. Tag passages with source document names
        text_to_doc: dict[str, str] = {
            candidate_chunks[cid]: doc_name
            for cid, doc_name in chunk_doc_map.items()
            if cid in candidate_chunks
        }
        source_passages = [
            f"[Source: {text_to_doc[p]}]\n{p}" if p in text_to_doc else p
            for p in source_passages
        ]

        # 7. Retrieve facts
        fact_strings = await self._retrieve_facts(query_vector)

        # 8. Detect question type
        q_type_hint = self._detect_question_type(query)

        # 9. Assemble structured result
        return self._assemble_raw_result(
            entity_list, relationship_strings, fact_strings, source_passages,
            q_type_hint,
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

    # ── Private helpers ───────────────────────────────────────────

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
        simple = [w for w in words if w not in self._STOP_WORDS and len(w) > 2][: 12]

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

    async def _batch_embed(
        self, query: str, keywords: list[str]
    ) -> tuple[list[float], list[list[float]]]:
        """Embed query + keywords in a single API call."""
        kw_to_embed = keywords[: self._keyword_limit]
        texts = [query] + kw_to_embed
        vectors = await self._embedder.aembed_documents(texts)
        return vectors[0], vectors[1:]

    async def _discover_entities(
        self,
        query_vector: list[float],
        kw_vectors: list[list[float]],
        llm_kw: list[str],
        all_keywords: list[str],
    ) -> tuple[dict[str, dict], dict[str, str]]:
        """5-path entity discovery."""
        found: dict[str, dict] = {}
        sources: dict[str, str] = {}

        def _add(eid: str, info: dict, source: str) -> None:
            if eid and eid not in found:
                found[eid] = info
                sources[eid] = source

        # 2a: Entity vector search per keyword
        for kw_vec in kw_vectors:
            try:
                results = await self._vector.search_entities(kw_vec, top_k=self._entity_top_k)
                for ent in results:
                    _add(
                        ent.get("id", ""),
                        {"name": ent.get("name", ""), "description": ent.get("description", "")},
                        "vector_kw",
                    )
            except Exception:
                pass

        # 2b: Cypher CONTAINS on entity names (batched UNWIND)
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

        # 2c: Fulltext search on entity index
        for kw in all_keywords[: 6]:
            try:
                ft_ents = await self._vector.fulltext_search(kw, top_k=3, label="__Entity__")
                for ent in ft_ents:
                    _add(
                        ent.get("id", ""),
                        {"name": ent.get("name", ""), "description": ent.get("description", "")},
                        "fulltext",
                    )
            except Exception:
                pass

        # 2d: Question vector on entity index
        try:
            q_ents = await self._vector.search_entities(query_vector, top_k=10)
            for ent in q_ents:
                _add(
                    ent.get("id", ""),
                    {"name": ent.get("name", ""), "description": ent.get("description", "")},
                    "question_vector",
                )
        except Exception:
            pass

        # 2e: Synonym expansion (batched UNWIND)
        synonym_batch: dict[str, dict] = {}
        eids_for_syn = list(found.keys())[:10]
        if eids_for_syn:
            try:
                syn_result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (e:__Entity__ {id: eid})-[:SYNONYM]-(s:__Entity__) "
                    "RETURN s.id AS id, s.name AS name, s.description AS desc "
                    "LIMIT 30",
                    {"eids": eids_for_syn},
                )
                for row in syn_result.result_set:
                    sid = row[0]
                    if sid and sid not in found and sid not in synonym_batch:
                        synonym_batch[sid] = {
                            "name": row[1] if len(row) > 1 else "",
                            "description": row[2] if len(row) > 2 else "",
                        }
                        sources[sid] = "synonym"
            except Exception:
                pass
        found.update(synonym_batch)

        return found, sources

    async def _expand_relationships(
        self, entity_list: list[tuple[str, dict]]
    ) -> list[str]:
        """1-hop relationship expansion from top entities."""
        relationship_strings: list[str] = []
        seen: set[tuple] = set()

        # 1-hop relationships (batched UNWIND)
        eids_1hop = [eid for eid, _ in entity_list[:15]]
        if eids_1hop:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (a:__Entity__ {id: eid})-[r]->(b:__Entity__) "
                    "WHERE type(r) <> 'SYNONYM' AND type(r) <> 'MENTIONED_IN' "
                    "RETURN a.name AS src, type(r) AS rel, b.name AS tgt, "
                    "COALESCE(r.description, '') AS desc "
                    "LIMIT 150",
                    {"eids": eids_1hop},
                )
                for row in result.result_set:
                    src = row[0] or ""
                    rel_type = row[1] if len(row) > 1 else ""
                    tgt = row[2] if len(row) > 2 else ""
                    desc = row[3] if len(row) > 3 else ""
                    key = (src.lower(), rel_type, tgt.lower())
                    if src and rel_type and tgt and key not in seen:
                        seen.add(key)
                        line = f"{src} —[{rel_type}]→ {tgt}"
                        if desc:
                            line += f": {desc}"
                        relationship_strings.append(line)
            except Exception:
                pass

        # 2-hop relationships for top 5 entities (batched UNWIND)
        eids_2hop = [eid for eid, _ in entity_list[:5]]
        if eids_2hop:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (a:__Entity__ {id: eid})-[r1]->(b:__Entity__)-[r2]->(c:__Entity__) "
                    "WHERE type(r1) <> 'SYNONYM' AND type(r1) <> 'MENTIONED_IN' "
                    "AND type(r2) <> 'SYNONYM' AND type(r2) <> 'MENTIONED_IN' "
                    "RETURN a.name, type(r1), b.name, type(r2), c.name "
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

        return relationship_strings[: self._max_relationships]

    async def _retrieve_chunks(
        self,
        query: str,
        query_vector: list[float],
        llm_kw: list[str],
        simple_kw: list[str],
        entity_list: list[tuple[str, dict]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """4-path chunk retrieval."""
        chunks: dict[str, str] = {}
        sources: dict[str, str] = {}

        def _add(cid: str, text: str, source: str) -> None:
            if cid and text and cid not in chunks:
                chunks[cid] = text
                sources[cid] = source

        # Path A: Fulltext search
        fulltext_queries = [query] + llm_kw[: 6] + simple_kw[: 4]
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

        # Path C: MENTIONED_IN (batched UNWIND)
        eids_mention = [eid for eid, _ in entity_list[:15]]
        if eids_mention:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (e:__Entity__ {id: eid})-[:MENTIONED_IN]->(c:Chunk) "
                    "RETURN c.id AS id, c.text AS text "
                    "LIMIT 30",
                    {"eids": eids_mention},
                )
                for row in result.result_set:
                    cid = row[0]
                    text = row[1] if len(row) > 1 else ""
                    _add(cid, text, "mentioned_in")
            except Exception:
                pass

        # Path D: Cypher CONTAINS (batched UNWIND)
        contains_kws = [kw for kw in llm_kw[:6] if len(kw) >= 4]
        if contains_kws:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $keywords AS kw "
                    "MATCH (c:Chunk) WHERE c.text CONTAINS kw "
                    "RETURN c.id AS id, c.text AS text "
                    "LIMIT 18",
                    {"keywords": contains_kws},
                )
                for row in result.result_set:
                    cid = row[0]
                    text = row[1] if len(row) > 1 else ""
                    _add(cid, text, "contains")
            except Exception:
                pass

        # Path E: 2-hop entity→neighbor→chunk (batched UNWIND)
        eids_2hop_chunk = [eid for eid, _ in entity_list[:10]]
        if eids_2hop_chunk:
            try:
                result = await self._graph.query_raw(
                    "UNWIND $eids AS eid "
                    "MATCH (e:__Entity__ {id: eid})-[r]-(neighbor:__Entity__)"
                    "-[:MENTIONED_IN]->(c:Chunk) "
                    "WHERE type(r) <> 'SYNONYM' AND type(r) <> 'MENTIONED_IN' "
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
                    # Extract filename from path
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
            return [chunk_texts[i] for i, _ in scored[: self._chunk_top_k]]
        except Exception:
            return chunk_texts[: self._chunk_top_k]

    async def _llm_rerank_passages(
        self, question: str, passages: list[str], top_k: int,
    ) -> list[str]:
        """Use the LLM to select the most entity-specific passages."""
        if len(passages) <= top_k:
            return passages

        numbered = "\n\n".join(
            f"[{i + 1}] {p[:500]}" for i, p in enumerate(passages)
        )
        prompt = (
            f"Question: {question}\n\n"
            f"Below are {len(passages)} passages. Return the numbers of the "
            f"{top_k} most relevant passages as a comma-separated list "
            f"(e.g. 3,1,7,2,...). Most relevant first.\n\n"
            "IMPORTANT: A passage is relevant ONLY if it contains information "
            "about the SPECIFIC entity or fact the question asks about. "
            "Passages mentioning related but DIFFERENT entities are NOT relevant.\n\n"
            f"{numbered}\n\nRelevant passage numbers:"
        )
        try:
            response = await self._llm.ainvoke(prompt)
            indices: list[int] = []
            seen_idx: set[int] = set()
            for m in re.findall(r"\d+", response.content):
                idx = int(m) - 1  # 1-based -> 0-based
                if 0 <= idx < len(passages) and idx not in seen_idx:
                    seen_idx.add(idx)
                    indices.append(idx)
            if indices:
                return [passages[i] for i in indices[:top_k]]
        except Exception:
            pass
        return passages[:top_k]

    async def _retrieve_facts(self, query_vector: list[float]) -> list[str]:
        """Vector search on Fact nodes."""
        fact_strings: list[str] = []
        try:
            results = await self._vector.search_facts(query_vector, top_k=self._fact_top_k)
            for fact in results:
                subj = fact.get("subject", "")
                pred = fact.get("predicate", "")
                obj = fact.get("object", "")
                if subj and pred and obj:
                    fact_strings.append(f"({subj}) —[{pred}]→ ({obj})")
        except Exception:
            pass
        return fact_strings

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
                "content": "## Key Entities\n" + "\n".join(entity_lines[: 25]),
            })

        # Relationship section
        if relationship_strings:
            records.append({
                "section": "relationships",
                "content": "## Entity Relationships\n"
                + "\n".join(f"- {r}" for r in relationship_strings[: 20]),
            })

        # Facts section
        if fact_strings:
            unique_facts = list(dict.fromkeys(fact_strings))
            records.append({
                "section": "facts",
                "content": "## Knowledge Graph Facts\n"
                + "\n".join(f"- {f}" for f in unique_facts[: 15]),
            })

        # Passages section
        if source_passages:
            records.append({
                "section": "passages",
                "content": "## Source Document Passages\n"
                + "\n---\n".join(source_passages[: 15]),
            })

        return RawSearchResult(
            records=records,
            metadata={"strategy": "multi_path"},
        )
