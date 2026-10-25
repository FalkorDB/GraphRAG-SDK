# GraphRAG SDK — Retrieval: Relationship Expansion
# Relevance-guided walk over RELATES edges from the top entities.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.retrieval.strategies.result_assembly import format_relation_line

logger = logging.getLogger(__name__)

# (entities expanded, edges kept per entity) for each round of the walk.
# Round 0 starts from the discovered entities; each later round expands the
# far ends of the closest edges found so far, so a chain question ("their
# student, then that student's child, then ...") is followed one link per
# round. The per-entity cap bounds what a hub entity with thousands of edges
# can return; the walk issues one query per round.
_ROUNDS: tuple[tuple[int, int], ...] = ((15, 10), (8, 5), (5, 5))

# Distance given to an edge with no embedding (written by finalize()), so it
# ranks after every embedded edge instead of being dropped. Above 2.0, the
# largest cosine distance, so it never ties an embedded edge.
_UNEMBEDDED_DISTANCE = 3.0

# Columns of one relation row, given the edge as ``r``. Direction and names
# come from the edge's endpoints: the walk matches edges in either direction,
# and a dedup merge leaves the stored src_name/tgt_name behind.
_RELATION_COLUMNS = (
    "startNode(r).name AS src, r.rel_type AS rel, endNode(r).name AS tgt, "
    "r.description AS description, r.fact AS fact"
)


def _distance_expr(edge: str, has_vector: bool) -> str:
    if not has_vector:
        return "0.0"
    return (
        f"CASE WHEN {edge}.embedding IS NULL THEN {_UNEMBEDDED_DISTANCE} "
        f"ELSE vec.cosineDistance({edge}.embedding, vecf32($qv)) END"
    )


async def expand_relationships(
    graph_store: Any,
    entity_list: list[tuple[str, dict]],
    max_relationships: int = 20,
    ctx: Context | None = None,
    query_vector: list[float] | None = None,
) -> list[str]:
    """Relationship expansion from the top entities, up to three links deep.

    Uses the single ``RELATES`` edge type. The ``rel_type`` property
    stores the original relationship type, and ``description`` the
    evidence. Edges are matched in both directions: an entity that is only
    ever the object of a relation ("X founded <entity>") has no outgoing
    edges at all.

    With ``query_vector``, edges are ranked by the cosine distance between
    the question and the edge embedding (written by ``finalize()``): each
    round keeps the closest edges per entity, the next round expands the far
    ends of the closest edges, and the closest edges overall are returned.
    Without it, each entity's edges are ordered by neighbour id and
    relationship type, and the lines are returned in the order reached.

    Returns:
        List of formatted relationship strings, closest first.
    """
    has_vector = query_vector is not None
    params_base: dict[str, Any] = {"qv": query_vector} if has_vector else {}
    scored: list[tuple[float, int, str]] = []  # (distance, order reached, line)
    seen: set[tuple[str, str, str]] = set()
    expanded: set[str] = set()
    frontier = [eid for eid, _ in entity_list]

    for round_no, (n_entities, per_entity) in enumerate(_ROUNDS):
        eids: list[str] = []
        for eid in frontier:
            if eid and eid not in expanded and eid not in eids:
                eids.append(eid)
            if len(eids) == n_entities:
                break
        if not eids:
            break
        expanded.update(eids)
        try:
            if ctx is not None:
                ctx.ensure_budget("relationship expansion")
            result = await graph_store.query_raw(
                "UNWIND $eids AS eid "
                "CALL { "
                "  WITH eid "
                "  MATCH (a:__Entity__ {id: eid})-[r:RELATES]-(b:__Entity__) "
                f"  WITH r, b, {_distance_expr('r', has_vector)} AS dist "
                "  RETURN r, b.id AS other, dist ORDER BY dist ASC, other ASC, r.rel_type ASC "
                f"  LIMIT {per_entity} "
                "} "
                f"RETURN {_RELATION_COLUMNS}, dist, other",
                {"eids": eids, **params_base},
            )
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.debug("Relationship expansion round %d failed: %s", round_no, exc)
            break

        reached: list[tuple[float, str]] = []
        for row in result.result_set:
            src = row[0] or ""
            rel_type = row[1] if len(row) > 1 and row[1] else ""
            tgt = row[2] if len(row) > 2 and row[2] else ""
            description = row[3] if len(row) > 3 and row[3] else ""
            fact = row[4] if len(row) > 4 and row[4] else ""
            dist = float(row[5]) if len(row) > 5 and row[5] is not None else _UNEMBEDDED_DISTANCE
            other = row[6] if len(row) > 6 and row[6] else ""
            if other:
                reached.append((dist, other))
            key = (src.lower(), rel_type, tgt.lower())
            if src and rel_type and tgt and key not in seen:
                seen.add(key)
                line = format_relation_line(src, rel_type, tgt, description, fact)
                scored.append((dist, len(scored), line))
        # Next round: the far ends of this round's closest edges.
        reached.sort()  # by distance, then id: ties break the same way every run
        frontier = [other for _, other in reached]

    if has_vector:
        # Equal distances break on the line itself: the order rows arrive in
        # is not guaranteed by the query.
        scored.sort(key=lambda s: (s[0], s[2]))
    return [line for _, _, line in scored[:max_relationships]]
