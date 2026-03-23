# GraphRAG SDK 2.0 — Retrieval: Chunk Retrieval
# 4-path chunk retrieval: fulltext + vector + MENTIONED_IN + 2-hop.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def retrieve_chunks(
    vector_store: Any,
    graph_store: Any,
    query: str,
    query_vector: list[float],
    llm_kw: list[str],
    simple_kw: list[str],
    entity_list: list[tuple[str, dict]],
) -> tuple[dict[str, str], dict[str, str]]:
    """4-path chunk retrieval: fulltext + vector + MENTIONED_IN + 2-hop.

    Returns:
        chunks: dict of chunk_id -> chunk_text
        sources: dict of chunk_id -> source path name
    """
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
            results = await vector_store.fulltext_search(ft_q, top_k=5, label="Chunk")
            for c in results:
                _add(c.get("id", ""), c.get("text", ""), "fulltext")
        except Exception:
            pass

    # Path B: Vector search
    try:
        results = await vector_store.search(query_vector, top_k=15, label="Chunk")
        for c in results:
            _add(c.get("id", ""), c.get("text", ""), "vector")
    except Exception:
        pass

    # Path C: MENTIONED_IN — 3 chunks per entity (batched UNWIND)
    eids_mention = [eid for eid, _ in entity_list[:15]]
    if eids_mention:
        try:
            result = await graph_store.query_raw(
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
            result = await graph_store.query_raw(
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


async def fetch_chunk_documents(
    graph_store: Any,
    chunk_ids: list[str],
) -> dict[str, str]:
    """Batch-fetch source document name for each chunk via PART_OF.

    Returns:
        Mapping of chunk_id -> document filename.
    """
    if not chunk_ids:
        return {}
    try:
        result = await graph_store.query_raw(
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
