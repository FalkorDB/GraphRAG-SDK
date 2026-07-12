# GraphRAG SDK — Tools: toolkit-owned graph queries and result conversion
# Parameterized Cypher only — user values NEVER interpolated into query text.

from __future__ import annotations

from typing import Any

from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.tools.models import CypherResult, DocumentRef, EntityCard, RelationTriple

_BULKY_PROPS = frozenset({"embedding", "source_chunk_ids"})


def _card_from_row(row: list[Any]) -> tuple[str, EntityCard]:
    """(id, name, description, labels, properties) row -> (id, EntityCard)."""
    eid, name, desc, labels, props = (list(row) + [None] * 5)[:5]
    label = next((str(lbl) for lbl in (labels or []) if lbl != "__Entity__"), "")
    clean_props = {
        k: v
        for k, v in (props or {}).items()
        if k not in _BULKY_PROPS and k not in ("id", "name", "description")
    }
    return str(eid), EntityCard(
        name=str(name or ""), label=label, description=desc or None, properties=clean_props
    )


async def enrich_entities(store: GraphStore, entity_ids: list[str]) -> dict[str, EntityCard]:
    """Fetch label + user properties for entity ids (bulky props excluded)."""
    if not entity_ids:
        return {}
    result = await store.query_raw(
        "UNWIND $ids AS eid MATCH (e:__Entity__ {id: eid}) "
        "RETURN e.id, e.name, e.description, labels(e), properties(e)",
        {"ids": entity_ids},
    )
    return dict(_card_from_row(row) for row in result.result_set)


async def expand_triples(
    store: GraphStore,
    seed_ids: list[str],
    *,
    hops: int,
    cap: int = 60,
    per_hop_limit: int = 25,
) -> list[RelationTriple]:
    """Frontier expansion over RELATES edges, hop by hop.

    Never builds a variable-length pattern from user input. Direction comes
    from the edge's own src_name/tgt_name properties, so one undirected
    query per hop suffices.
    """
    triples: list[RelationTriple] = []
    seen: set[tuple[str, str, str]] = set()
    visited = set(seed_ids)
    frontier = list(seed_ids)
    for _ in range(max(1, hops)):
        if not frontier or len(triples) >= cap:
            break
        result = await store.query_raw(
            "MATCH (a:__Entity__)-[r:RELATES]-(b:__Entity__) "
            "WHERE a.id IN $ids "
            "RETURN r.src_name, r.rel_type, r.tgt_name, "
            "COALESCE(r.fact, r.description, ''), b.id "
            "ORDER BY r.src_name, r.rel_type, r.tgt_name "
            "LIMIT $limit",
            {"ids": frontier, "limit": per_hop_limit},
        )
        next_frontier: list[str] = []
        for row in result.result_set:
            src, rel_type, tgt = row[0] or "", row[1] or "", row[2] or ""
            fact = row[3] if len(row) > 3 else ""
            other = row[4] if len(row) > 4 else ""
            key = (src.lower(), rel_type, tgt.lower())
            if src and rel_type and tgt and key not in seen:
                seen.add(key)
                triples.append(
                    RelationTriple(source=src, type=rel_type, target=tgt, fact=fact or None)
                )
            if other and other not in visited:
                visited.add(other)
                next_frontier.append(other)
        frontier = next_frontier
    return triples[:cap]


async def chunk_documents(store: GraphStore, chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
    """chunk_id -> (document_id, document_path) via PART_OF."""
    if not chunk_ids:
        return {}
    result = await store.query_raw(
        "UNWIND $ids AS cid MATCH (d:Document)-[:PART_OF]->(c:Chunk {id: cid}) "
        "RETURN c.id, d.id, d.path",
        {"ids": chunk_ids},
    )
    return {row[0]: (str(row[1] or ""), str(row[2] or "")) for row in result.result_set if row[0]}


async def find_entity_matches(
    store: GraphStore, name: str, limit: int = 5
) -> list[tuple[str, EntityCard]]:
    """Ranked (exact > case-insensitive > substring; shorter first) name matches."""
    result = await store.query_raw(
        "MATCH (e:__Entity__) WHERE toLower(e.name) CONTAINS toLower($name) "
        "RETURN e.id, e.name, e.description, labels(e), properties(e), "
        "CASE WHEN e.name = $name THEN 0 "
        "WHEN toLower(e.name) = toLower($name) THEN 1 ELSE 2 END AS rank "
        "ORDER BY rank, size(e.name), e.name LIMIT $limit",
        {"name": name, "limit": limit},
    )
    return [_card_from_row(list(row)[:5]) for row in result.result_set]


async def entity_documents(store: GraphStore, entity_id: str) -> list[DocumentRef]:
    """Distinct source documents mentioning the entity."""
    result = await store.query_raw(
        "MATCH (e:__Entity__ {id: $eid})-[:MENTIONED_IN]->(:Chunk)"
        "<-[:PART_OF]-(d:Document) "
        "RETURN DISTINCT d.id, d.path ORDER BY d.id LIMIT 10",
        {"eid": entity_id},
    )
    return [
        DocumentRef(document_id=str(row[0] or ""), document_path=str(row[1] or ""))
        for row in result.result_set
        if row[0]
    ]


async def schema_counts(store: GraphStore) -> tuple[dict[str, int], dict[str, int]]:
    """(entity-label -> count, RELATES rel_type -> count) from the live graph."""
    labels = await store.query_raw(
        "MATCH (n:__Entity__) UNWIND labels(n) AS l "
        "WITH l, count(*) AS c WHERE l <> '__Entity__' "
        "RETURN l, c ORDER BY l"
    )
    rels = await store.query_raw(
        "MATCH (:__Entity__)-[r:RELATES]->(:__Entity__) "
        "WITH r.rel_type AS t, count(*) AS c WHERE t IS NOT NULL "
        "RETURN t, c ORDER BY t"
    )
    return (
        {row[0]: int(row[1]) for row in labels.result_set},
        {row[0]: int(row[1]) for row in rels.result_set},
    )


def _to_jsonable(value: Any) -> Any:
    """Convert falkordb values to JSON-safe data; strips bulky/binary props.

    falkordb attribute contract (verified against falkordb-py 1.x):
    ``Node{labels, properties}``, ``Edge{relation, src_node, dest_node,
    properties}``. Duck-typed so plain scalars/containers pass through.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return GraphStore._sanitize_string(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        clean = {k: _to_jsonable(v) for k, v in props.items() if k not in _BULKY_PROPS}
        labels = getattr(value, "labels", None)
        if labels is not None:
            return {"labels": [str(lbl) for lbl in labels], "properties": clean}
        relation = getattr(value, "relation", None)
        if relation is not None:
            return {
                "type": str(relation),
                "src": getattr(value, "src_node", None),
                "dst": getattr(value, "dest_node", None),
                "properties": clean,
            }
    return GraphStore._sanitize_string(str(value))


def convert_query_result(result: Any, *, limit: int, limit_injected: bool) -> CypherResult:
    """FalkorDB QueryResult -> CypherResult (columns from header pairs)."""
    header = getattr(result, "header", None) or []
    columns = [
        str(h[1]) if isinstance(h, (list, tuple)) and len(h) >= 2 else str(h) for h in header
    ]
    rows = [[_to_jsonable(v) for v in row] for row in (result.result_set or [])]
    return CypherResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=bool(limit_injected and len(rows) == limit),
    )
