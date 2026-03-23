# GraphRAG SDK 2.0 — Retrieval: Relationship Expansion
# 1-hop + 2-hop relationship traversal from top entities.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def expand_relationships(
    graph_store: Any,
    entity_list: list[tuple[str, dict]],
    max_relationships: int = 20,
) -> list[str]:
    """1-hop + 2-hop relationship expansion from top entities.

    Uses the single ``RELATES`` edge type. The ``rel_type`` property
    stores the original relationship type, and ``fact`` stores the
    evidence.

    Returns:
        List of formatted relationship strings.
    """
    relationship_strings: list[str] = []
    seen: set[tuple] = set()

    # 1-hop relationships (batched UNWIND)
    eids_1hop = [eid for eid, _ in entity_list[:15]]
    if eids_1hop:
        try:
            result = await graph_store.query_raw(
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
        except Exception as exc:
            logger.debug("Relationship expansion failed: %s", exc)

    # 2-hop relationships for top 5 entities (batched UNWIND)
    eids_2hop = [eid for eid, _ in entity_list[:5]]
    if eids_2hop:
        try:
            result = await graph_store.query_raw(
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
        except Exception as exc:
            logger.debug("Relationship expansion failed: %s", exc)

    return relationship_strings[:max_relationships]
