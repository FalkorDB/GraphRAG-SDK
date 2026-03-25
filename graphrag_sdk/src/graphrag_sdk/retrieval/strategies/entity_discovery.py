# GraphRAG SDK 2.0 — Retrieval: Entity Discovery
# 2-path entity discovery: Cypher CONTAINS + fulltext search.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def search_relates_edges(
    vector_store: Any,
    query_vector: list[float],
    rel_top_k: int = 15,
) -> tuple[list[tuple[str, float]], dict[str, dict]]:
    """Search RELATES edges by vector similarity.

    Returns:
        fact_strings_with_scores: [("src -[type]-> tgt: fact_text", score), ...]
        entities: dict of entity_id -> {name, description} discovered
                  from matched edge endpoints.
    """
    fact_strings: list[tuple[str, float]] = []
    entities: dict[str, dict] = {}
    try:
        results = await vector_store.search_relationships(query_vector, top_k=rel_top_k)
        for rel in results:
            src = rel.get("src_name", "")
            tgt = rel.get("tgt_name", "")
            rel_type = rel.get("type", "")
            fact = rel.get("fact", "")
            score = float(rel.get("score", 0.0))
            if src and rel_type and tgt:
                line = f"{src} —[{rel_type}]→ {tgt}"
                if fact:
                    line += f": {fact}"
                fact_strings.append((line, score))
            # Add entities as graph entry points
            if src:
                src_id = src.strip().lower().replace(" ", "_")
                if src_id not in entities:
                    entities[src_id] = {"name": src, "description": ""}
            if tgt:
                tgt_id = tgt.strip().lower().replace(" ", "_")
                if tgt_id not in entities:
                    entities[tgt_id] = {"name": tgt, "description": ""}
    except Exception as exc:
        logger.debug("RELATES edge vector search failed: %s", exc)
    return fact_strings, entities


async def discover_entities(
    graph_store: Any,
    vector_store: Any,
    llm_kw: list[str],
    all_keywords: list[str],
) -> tuple[dict[str, dict], dict[str, str]]:
    """2-path entity discovery.

    Paths:
    a: Cypher CONTAINS on entity names
    b: Fulltext search on entity index

    Returns:
        found: dict of entity_id -> {name, description}
        sources: dict of entity_id -> source path name
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
            result = await graph_store.query_raw(
                "UNWIND $keywords AS kw "
                "MATCH (e:__Entity__) WHERE toLower(e.name) CONTAINS toLower(kw) "
                "RETURN e.id AS id, e.name AS name, e.description AS desc "
                "LIMIT 40",
                {"keywords": kw_batch},
            )
            for row in result.result_set:
                _add(
                    row[0],
                    {
                        "name": row[1] if len(row) > 1 else "",
                        "description": row[2] if len(row) > 2 else "",
                    },
                    "cypher_contains",
                )
        except Exception as exc:
            logger.debug("Entity CONTAINS search failed: %s", exc)

    # Path b: Fulltext search on entity index
    for kw in all_keywords[:6]:
        try:
            ft_ents = await vector_store.fulltext_search(kw, top_k=3, label="__Entity__")
            for ent in ft_ents:
                eid = ent.get("id", "")
                if eid:
                    try:
                        detail = await graph_store.query_raw(
                            "MATCH (e:__Entity__ {id: $eid}) "
                            "RETURN e.name AS name, e.description AS desc",
                            {"eid": eid},
                        )
                        if detail.result_set:
                            row = detail.result_set[0]
                            _add(
                                eid,
                                {
                                    "name": row[0] if row[0] else "",
                                    "description": row[1] if len(row) > 1 and row[1] else "",
                                },
                                "fulltext",
                            )
                    except Exception:
                        logger.debug("Entity detail fetch failed for %s", eid, exc_info=True)
                        _add(eid, {"name": "", "description": ""}, "fulltext")
        except Exception as exc:
            logger.debug("Entity fulltext search failed for '%s': %s", kw, exc)

    return found, sources
