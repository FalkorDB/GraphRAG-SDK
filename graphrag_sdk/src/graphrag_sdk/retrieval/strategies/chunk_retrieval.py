# GraphRAG SDK — Retrieval: Chunk Retrieval
# 4-path chunk retrieval: fulltext + vector + MENTIONED_IN + 2-hop.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError

logger = logging.getLogger(__name__)

# Distance of chunk ``c`` from the question; a chunk with no embedding (a
# structured record) ranks after every embedded one instead of being dropped.
_CHUNK_DISTANCE = (
    "CASE WHEN c.embedding IS NULL THEN 2.0 ELSE vec.cosineDistance(c.embedding, vecf32($qv)) END"
)


async def retrieve_chunks(
    vector_store: Any,
    graph_store: Any,
    query: str,
    query_vector: list[float],
    llm_kw: list[str],
    simple_kw: list[str],
    entity_list: list[tuple[str, dict]],
    ctx: Context | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, list[float]]]:
    """4-path chunk retrieval: fulltext + vector + MENTIONED_IN + 2-hop.

    Returns:
        chunks: dict of chunk_id -> chunk_text
        sources: dict of chunk_id -> source path name
        embeddings: dict of chunk_id -> stored embedding vector
    """
    chunks: dict[str, str] = {}
    sources: dict[str, str] = {}
    texts: set[str] = set()

    def _add(cid: str, text: str, source: str) -> None:
        # The same text under two ids (an update in flight, or one passage
        # ingested twice) is one passage to the answer.
        if cid and text and cid not in chunks and text not in texts:
            chunks[cid] = text
            sources[cid] = source
            texts.add(text)

    # Path A: Fulltext search
    fulltext_queries = [query] + llm_kw[:6] + simple_kw[:4]
    for ft_q in fulltext_queries:
        try:
            if ctx is not None:
                ctx.ensure_budget("chunk fulltext search")
            results = await vector_store.fulltext_search_chunks(ft_q, top_k=5)
            for c in results:
                _add(c.get("id", ""), c.get("text", ""), "fulltext")
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.debug("Chunk fulltext search failed for query: %s", exc)

    # Path B: Vector search
    try:
        if ctx is not None:
            ctx.ensure_budget("chunk vector search")
        results = await vector_store.search_chunks(query_vector, top_k=15)
        for c in results:
            _add(c.get("id", ""), c.get("text", ""), "vector")
    except LatencyBudgetExceededError:
        raise
    except Exception as exc:
        logger.debug("Chunk vector search failed: %s", exc)

    # Path C: MENTIONED_IN — top-3 chunks per entity, ranked by cosine
    # distance to the query embedding. Hub entities (e.g. the main
    # product name) can be MENTIONED_IN hundreds of chunks; the previous
    # COLLECT(c)[..3] picked an arbitrary 3, almost never including the
    # chunks most relevant to the current query. Ranking by cosine here
    # surfaces the chunks closest to the query intent. Chunks without an
    # embedding are kept, after the embedded ones: structured-record chunks
    # are never embedded and are reached only through their entities.
    eids_mention = [eid for eid, _ in entity_list[:15]]
    if eids_mention:
        try:
            if ctx is not None:
                ctx.ensure_budget("MENTIONED_IN chunk retrieval")
            result = await graph_store.query_raw(
                "UNWIND $eids AS eid "
                "MATCH (e:__Entity__ {id: eid})-[:MENTIONED_IN]->(c:Chunk) "
                f"WITH eid, c, {_CHUNK_DISTANCE} AS dist "
                "ORDER BY eid, dist ASC, c.id ASC "
                "WITH eid, COLLECT(c)[..3] AS chunks "
                "UNWIND chunks AS c "
                "RETURN eid, c.id AS id, c.text AS text",
                {"eids": eids_mention, "qv": query_vector},
            )
            for row in result.result_set:
                cid = row[1]
                text = row[2] if len(row) > 2 else ""
                _add(cid, text, "mentioned_in")
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.debug("MENTIONED_IN chunk retrieval failed: %s", exc)

    # Path D: 2-hop entity→neighbor→chunk (batched UNWIND), the closest 20
    # to the question; without an order the LIMIT kept an arbitrary 20.
    eids_2hop_chunk = [eid for eid, _ in entity_list[:10]]
    if eids_2hop_chunk:
        try:
            if ctx is not None:
                ctx.ensure_budget("2-hop chunk retrieval")
            result = await graph_store.query_raw(
                "UNWIND $eids AS eid "
                "MATCH (e:__Entity__ {id: eid})-[:RELATES]-(neighbor:__Entity__)"
                "-[:MENTIONED_IN]->(c:Chunk) "
                f"WITH DISTINCT c, {_CHUNK_DISTANCE} AS dist "
                "RETURN c.id AS id, c.text AS text "
                "ORDER BY dist ASC, id ASC "
                "LIMIT 20",
                {"eids": eids_2hop_chunk, "qv": query_vector},
            )
            for row in result.result_set:
                cid = row[0]
                text = row[1] if len(row) > 1 else ""
                _add(cid, text, "2hop_mentioned")
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.debug("2-hop chunk retrieval failed: %s", exc)

    # Batch-fetch stored embeddings for all collected chunks
    embeddings: dict[str, list[float]] = {}
    missing_ids = list(chunks.keys())
    if missing_ids:
        try:
            if ctx is not None:
                ctx.ensure_budget("stored chunk embedding fetch")
            result = await graph_store.query_raw(
                "UNWIND $ids AS cid "
                "MATCH (c:Chunk {id: cid}) "
                "WHERE c.embedding IS NOT NULL "
                "RETURN c.id, c.embedding",
                {"ids": missing_ids},
            )
            for row in result.result_set:
                if row[0] and row[1] is not None:
                    embeddings[row[0]] = list(row[1])
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.debug("Stored embedding fetch failed: %s", exc)

    return chunks, sources, embeddings


async def fetch_chunk_documents(
    graph_store: Any,
    chunk_ids: list[str],
    ctx: Context | None = None,
) -> dict[str, str]:
    """Batch-fetch the source document path for each chunk via PART_OF.

    Returns the value of ``Document.path`` exactly as it was stored at
    ingestion time — typically a path relative to the ingestion root
    (e.g. ``"operations/falkordblite/falkordblite-py.md"``). Downstream
    consumers (citation rendering, source-link builders) need the full
    relative path; basenames alone are ambiguous when the same filename
    appears in multiple directories (e.g. ``index.md``).

    Returns:
        Mapping of chunk_id -> document path.
    """
    if not chunk_ids:
        return {}
    try:
        if ctx is not None:
            ctx.ensure_budget("chunk source document fetch")
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
                mapping[cid] = path
        return mapping
    except LatencyBudgetExceededError:
        raise
    except Exception as exc:
        logger.debug("Document name fetch failed: %s", exc)
        return {}


async def inactive_chunk_ids(
    graph_store: Any,
    chunk_ids: list[str],
    ctx: Context | None = None,
) -> set[str]:
    """Chunks whose Document is not live: being deleted, or the
    ``<id>__pending__<hex>`` copy an ``update()`` writes before cutover.

    Until cutover the old version stays live, so the pending copy would show
    the same passages twice, or a half-written new version; a document
    marked ``pending_delete`` is on its way out.
    """
    if not chunk_ids:
        return set()
    try:
        if ctx is not None:
            ctx.ensure_budget("chunk document state fetch")
        result = await graph_store.query_raw(
            "UNWIND $ids AS cid "
            "MATCH (d:Document)-[:PART_OF]->(c:Chunk {id: cid}) "
            "WHERE d.id CONTAINS '__pending__' OR d.pending_delete = true "
            "RETURN DISTINCT c.id",
            {"ids": chunk_ids},
        )
        return {row[0] for row in result.result_set if row and row[0]}
    except LatencyBudgetExceededError:
        raise
    except Exception as exc:
        logger.debug("Chunk document state fetch failed: %s", exc)
        return set()
