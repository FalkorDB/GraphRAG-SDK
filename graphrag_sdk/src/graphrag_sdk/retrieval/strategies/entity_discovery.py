# GraphRAG SDK — Retrieval: Entity Discovery
# 2-path entity discovery: Cypher CONTAINS + fulltext search,
# with optional sibling expansion for enumeration queries.

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_ENUMERATION_RE = re.compile(
    r"\b(every|each|complete list|full list|list all|list of all"
    r"|enumerate|name all|name every|all the|all of the)\b",
    re.IGNORECASE,
)


def is_enumeration_query(query: str) -> bool:
    """Return True if the query asks to list/enumerate all items."""
    return bool(_ENUMERATION_RE.search(query))


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

    # Path a: Cypher name match (exact-first, then CONTAINS). Both use a
    # per-keyword quota so a broad keyword (e.g. "FalkorDB") cannot starve
    # rare keywords (e.g. "indegree") out of downstream entity-list caps,
    # even when the graph has many homonyms of the broad term.
    seen_kw: set[str] = set()
    kw_batch: list[str] = []
    for kw in llm_kw[:8]:
        if not kw:
            continue
        k_low = kw.lower()
        if k_low in seen_kw:
            continue
        seen_kw.add(k_low)
        kw_batch.append(kw)

    if kw_batch:
        # Pass a1: exact-name matches, capped per keyword. Inserted first
        # so exact matches land at the head of `found` and survive the
        # downstream max_entities / result_assembly caps.
        try:
            result = await graph_store.query_raw(
                "UNWIND $keywords AS kw "
                "CALL { "
                "  WITH kw "
                "  MATCH (e:__Entity__) WHERE toLower(e.name) = toLower(kw) "
                "  RETURN e.id AS id, e.name AS name, e.description AS desc "
                "  LIMIT 3 "
                "} "
                "RETURN id, name, desc",
                {"keywords": kw_batch},
            )
            for row in result.result_set:
                _add(
                    row[0],
                    {
                        "name": row[1] if len(row) > 1 else "",
                        "description": row[2] if len(row) > 2 else "",
                    },
                    "cypher_exact",
                )
        except Exception as exc:
            logger.debug("Entity exact-name search failed: %s", exc)

        # Pass a2: CONTAINS with per-keyword quota and shorter-name priority.
        # Excludes exact matches (already added in pass a1) so the quota
        # isn't spent re-fetching them.
        try:
            result = await graph_store.query_raw(
                "UNWIND $keywords AS kw "
                "CALL { "
                "  WITH kw "
                "  MATCH (e:__Entity__) "
                "  WHERE toLower(e.name) CONTAINS toLower(kw) "
                "    AND toLower(e.name) <> toLower(kw) "
                "  RETURN e.id AS id, e.name AS name, e.description AS desc "
                "  ORDER BY size(e.name) ASC "
                "  LIMIT 5 "
                "} "
                "RETURN id, name, desc",
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


async def expand_sibling_entities(
    graph_store: Any,
    found_entities: dict[str, dict],
    found_sources: dict[str, str],
    max_siblings: int = 20,
) -> int:
    """Expand discovered entities by finding graph siblings.

    Finds hub entities (``__Entity__`` nodes connected to 2+ already-
    discovered entities), then returns their other ``__Entity__``
    neighbours.  This catches structurally related entities (e.g.
    ``list.remove`` when ``list.dedup``, ``list.insert``, ``list.sort``
    are already found) that may have been missed by vector similarity.

    Mutates *found_entities* and *found_sources* in place.
    Returns the number of new entities added.
    """
    if len(found_entities) < 2:
        return 0

    found_ids = list(found_entities.keys())
    added = 0

    try:
        result = await graph_store.query_raw(
            "MATCH (e:__Entity__) WHERE e.id IN $found_ids "
            "MATCH (e)-[]-(hub:__Entity__) "
            "WITH hub, collect(DISTINCT e.id) AS via "
            "WHERE size(via) >= 2 "
            "MATCH (hub)-[]-(sibling:__Entity__) "
            "WHERE NOT sibling.id IN $found_ids "
            "RETURN DISTINCT sibling.id AS id, sibling.name AS name, "
            "sibling.description AS desc "
            "ORDER BY sibling.name "
            "LIMIT $limit",
            {"found_ids": found_ids, "limit": max_siblings},
        )
        for row in result.result_set:
            eid = row[0]
            if eid and eid not in found_entities:
                found_entities[eid] = {
                    "name": row[1] if len(row) > 1 else "",
                    "description": row[2] if len(row) > 2 else "",
                }
                found_sources[eid] = "sibling_expansion"
                added += 1
    except Exception as exc:
        logger.debug("Sibling entity expansion failed: %s", exc)

    return added
