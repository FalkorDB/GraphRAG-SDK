# GraphRAG SDK — Tools: graph queries behind the toolkit
# Parameterized Cypher only: caller values are never put into query text.
# Every query runs read-only (GRAPH.RO_QUERY) with a database timeout.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.storage import system_store
from graphrag_sdk.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)

#: Properties never returned to a caller: vectors and bulky bookkeeping.
_BULKY_PROPS = frozenset({"source_chunk_ids", "spans"})
#: Keys of the SDK's ``__GraphRAGConfig__`` node (never shown).
_CONFIG_KEYS = frozenset({"sdk_version", "embedding_dimension"})
#: ``kind`` values of the SDK's ``__GraphRAGProcessing__`` records.
_PROCESSING_KINDS = frozenset(
    {
        system_store.RUN,
        system_store.PENDING,
        system_store.UNRESOLVED,
        system_store.DELETED,
        system_store.PLAN_HEADER,
        system_store.PLAN,
        system_store.PLAN_IDS,
    }
)
#: A list of at least this many numbers is treated as a vector and omitted.
VECTOR_MIN_LENGTH = 64
#: Longest string returned in one cell.
MAX_CELL_CHARS = 2000


class NoGraph:
    """The result of reading a graph that does not exist yet: no rows."""

    header: list[Any] = []
    result_set: list[Any] = []


def _is_missing_graph(exc: Exception) -> bool:
    # FalkorDB answers any query on a graph key that was never written with
    # "Invalid graph operation on empty key"; for a read that is just no rows.
    return "empty key" in str(exc)


async def graph_exists(store: Any) -> bool | None:
    """Whether the graph key exists: ``False`` only when FalkorDB says it was
    never written, ``None`` when another error leaves it unknown. Read-only,
    so asking never creates the graph."""
    try:
        await store.query_raw("RETURN 1", {}, read_only=True)
    except Exception as exc:
        return False if _is_missing_graph(exc) else None
    return True


async def _read(store: Any, cypher: str, params: dict[str, Any] | None = None) -> Any:
    try:
        return await store.query_raw(cypher, params or {}, read_only=True)
    except Exception as exc:
        if _is_missing_graph(exc):
            return NoGraph()
        raise


def _rows(result: Any) -> list[Any]:
    return list(getattr(result, "result_set", None) or [])


# ── Values ───────────────────────────────────────────────────────


def _is_vector(value: Any) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) >= VECTOR_MIN_LENGTH
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    )


def _is_system_map(value: dict) -> bool:
    """Whether a property map (``properties(n)``, ``n {.*}``) is a system node's.

    The guard rejects system labels named in the query, but a label built at
    run time slips past it, so their maps are recognised by shape: no entity
    ``name``, and the config node's keys or a processing record's ``kind``.
    """
    if "name" in value:
        return False
    if _CONFIG_KEYS & set(value):
        return True
    kind = value.get("kind")
    return isinstance(kind, str) and kind in _PROCESSING_KINDS


def _public_labels(labels: Any) -> list[str]:
    return [str(lbl) for lbl in labels or [] if not str(lbl).startswith("__")]


def _path_text(path: Any, max_chars: int) -> str:
    """A path as ``(Alice)-[WORKS_AT]->(Acme)``: node names and relation
    types, each edge drawn in its stored direction, no properties."""
    nodes, edges = list(path.nodes()), list(path.edges())

    def name(node: Any) -> str:
        labels = getattr(node, "labels", None) or []
        if any(system_store.is_system_label(lbl) for lbl in labels):
            return "internal node"
        props = getattr(node, "properties", None) or {}
        label = (_public_labels(labels) or [""])[0]
        return str(props.get("name") or props.get("id") or label or "?")

    def node_id(end: Any) -> Any:
        return getattr(end, "id", end)

    text = f"({name(nodes[0])})" if nodes else ""
    for i, edge in enumerate(edges):
        rel = (getattr(edge, "properties", None) or {}).get("rel_type") or edge.relation
        here = nodes[i] if i < len(nodes) else None
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None
        forward = here is None or node_id(edge.src_node) == node_id(here)
        arrow = f"-[{rel}]->" if forward else f"<-[{rel}]-"
        text += arrow + (f"({name(nxt)})" if nxt is not None else "()")
    return to_jsonable(text, max_chars=max_chars)


def to_jsonable(value: Any, *, max_chars: int = MAX_CELL_CHARS) -> Any:
    """Convert a FalkorDB result value to JSON-safe data a model may read.

    Nodes and edges become property maps (with labels / type). Embedding
    vectors, bulky bookkeeping properties and the SDK's internal config node
    are never returned, however they were reached; long strings are cut.
    """
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        text = GraphStore._sanitize_string(value)
        return text if len(text) <= max_chars else text[:max_chars] + "…[truncated]"
    if _is_vector(value):
        return f"[vector of {len(value)} numbers omitted]"
    if isinstance(value, list | tuple):
        return [to_jsonable(v, max_chars=max_chars) for v in value]
    if callable(getattr(value, "nodes", None)) and callable(getattr(value, "edges", None)):
        return _path_text(value, max_chars)
    if isinstance(value, dict):
        if _is_system_map(value):
            return "[internal node omitted]"
        return {
            str(k): to_jsonable(v, max_chars=max_chars)
            for k, v in value.items()
            if "embedding" not in str(k).lower() and k not in _BULKY_PROPS
        }
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        labels = getattr(value, "labels", None)
        if labels is not None and any(system_store.is_system_label(lbl) for lbl in labels):
            return "[internal node omitted]"
        clean = {
            k: to_jsonable(v, max_chars=max_chars)
            for k, v in props.items()
            if "embedding" not in k.lower() and k not in _BULKY_PROPS and not _is_vector(v)
        }
        if labels is not None:
            return {"labels": _public_labels(labels), **clean}
        relation = getattr(value, "relation", None)
        if relation is not None:
            return {"type": str(relation), **clean}
        return clean
    return GraphStore._sanitize_string(str(value))


# ── Status / ontology ────────────────────────────────────────────


async def schema_counts(store: Any) -> tuple[dict[str, int], dict[str, int]]:
    """(entity label -> count, RELATES rel_type -> count) from the live graph."""
    labels = await _read(
        store,
        "MATCH (n:__Entity__) UNWIND labels(n) AS l "
        "WITH l, count(*) AS c WHERE NOT l STARTS WITH '__' "
        "RETURN l, c ORDER BY l",
    )
    rels = await _read(
        store,
        "MATCH (:__Entity__)-[r:RELATES]->(:__Entity__) "
        "WITH r.rel_type AS t, count(*) AS c WHERE t IS NOT NULL "
        "RETURN t, c ORDER BY t",
    )
    return (
        {str(row[0]): int(row[1]) for row in _rows(labels)},
        {str(row[0]): int(row[1]) for row in _rows(rels)},
    )


async def graph_counts(store: Any) -> dict[str, int]:
    """Entity, relation, document and chunk counts."""
    out: dict[str, int] = {}
    for key, cypher in (
        ("entities", "MATCH (n:__Entity__) RETURN count(n)"),
        ("relations", "MATCH (:__Entity__)-[r:RELATES]->(:__Entity__) RETURN count(r)"),
        ("documents", "MATCH (d:Document) RETURN count(d)"),
        ("chunks", "MATCH (c:Chunk) RETURN count(c)"),
    ):
        rows = _rows(await _read(store, cypher))
        out[key] = int(rows[0][0]) if rows and rows[0] and rows[0][0] is not None else 0
    return out


# ── Retrieval enrichment ─────────────────────────────────────────


async def enrich_entities(store: Any, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
    """id -> ``{id, name, label, description, properties}`` (no vectors)."""
    if not entity_ids:
        return {}
    result = await _read(
        store,
        "UNWIND $ids AS eid MATCH (e:__Entity__ {id: eid}) "
        "RETURN e.id, e.name, e.description, labels(e), properties(e)",
        {"ids": entity_ids},
    )
    cards: dict[str, dict[str, Any]] = {}
    for row in _rows(result):
        eid, name, desc, labels, props = (list(row) + [None] * 5)[:5]
        clean = to_jsonable(props or {}, max_chars=300)
        if isinstance(clean, dict):
            for key in ("id", "name", "description"):
                clean.pop(key, None)
        cards[str(eid)] = {
            "id": str(eid),
            "name": str(name or ""),
            "label": next(iter(_public_labels(labels)), ""),
            "description": str(desc) if desc else None,
            "properties": clean if isinstance(clean, dict) else {},
        }
    return cards


def fact_body(fact: str, rel_type: str) -> str:
    """``fact`` without the ``(source, TYPE, target): `` prefix extraction adds.

    The caller already shows source, type and target, so the prefix only
    repeats them. A fact that is nothing but the triple becomes ``""``.
    """
    fact = str(fact or "").strip()
    if not fact.startswith("(") or not rel_type:
        return fact
    at = fact.find(f", {rel_type}, ")
    if at < 0:
        return fact
    end = fact.find("): ", at)
    if end >= 0:
        return fact[end + 3 :].strip()
    if fact.endswith(")") and fact.find(")", at) == len(fact) - 1:
        return ""
    return fact


async def expand_triples(
    store: Any,
    seed_ids: list[str],
    *,
    hops: int,
    cap: int = 60,
    per_hop_limit: int = 25,
) -> list[dict[str, str]]:
    """``{source, type, target, fact}`` RELATES edges around ``seed_ids``, hop by hop.

    Never builds a variable-length pattern from caller input. Direction comes
    from the edge's own ``src_name`` / ``tgt_name``, so one undirected query
    per hop is enough. Edges an earlier hop returned are excluded before the
    ``LIMIT``, so walking back along them cannot use up a hop's budget.
    """
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    seen_edges: list[int] = []
    visited = set(seed_ids)
    frontier = list(seed_ids)
    for _ in range(max(1, hops)):
        if not frontier or len(triples) >= cap:
            break
        result = await _read(
            store,
            "MATCH (a:__Entity__)-[r:RELATES]-(b:__Entity__) "
            "WHERE a.id IN $ids AND NOT ID(r) IN $seen_edges "
            "RETURN r.src_name, r.rel_type, r.tgt_name, "
            "coalesce(r.fact, r.description, ''), b.id, ID(r) "
            "ORDER BY r.src_name, r.rel_type, r.tgt_name "
            "LIMIT $limit",
            {"ids": frontier, "seen_edges": list(seen_edges), "limit": per_hop_limit},
        )
        next_frontier: list[str] = []
        for row in _rows(result):
            src, rel_type, tgt = row[0] or "", row[1] or "", row[2] or ""
            fact = row[3] if len(row) > 3 else ""
            other = row[4] if len(row) > 4 else ""
            if len(row) > 5 and isinstance(row[5], int):
                seen_edges.append(row[5])
            key = (str(src).lower(), str(rel_type), str(tgt).lower())
            if src and rel_type and tgt and key not in seen:
                seen.add(key)
                triples.append(
                    {
                        "source": str(src),
                        "type": str(rel_type),
                        "target": str(tgt),
                        "fact": fact_body(str(fact or ""), str(rel_type)),
                    }
                )
            if other and other not in visited:
                visited.add(other)
                next_frontier.append(other)
        frontier = next_frontier
    return triples[:cap]


async def chunk_documents(store: Any, chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
    """chunk_id -> (document_id, document_path) via PART_OF."""
    if not chunk_ids:
        return {}
    result = await _read(
        store,
        "UNWIND $ids AS cid MATCH (d:Document)-[:PART_OF]->(c:Chunk {id: cid}) "
        "RETURN c.id, d.id, d.path",
        {"ids": chunk_ids},
    )
    return {str(row[0]): (str(row[1] or ""), str(row[2] or "")) for row in _rows(result) if row[0]}
