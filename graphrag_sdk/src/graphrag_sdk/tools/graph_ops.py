# GraphRAG SDK — Tools: graph queries behind the toolkit
# Parameterized Cypher only: caller values are never put into query text.
# Every query runs read-only (GRAPH.RO_QUERY) with a database timeout.

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.storage.graph_store import GraphStore
from graphrag_sdk.tools.cypher_guard import INTERNAL_NODE_DENYLIST, STRUCTURAL_RELATIONSHIPS
from graphrag_sdk.tools.errors import ToolError

logger = logging.getLogger(__name__)

#: ``asyncio.wait_for`` raises ``asyncio.TimeoutError``, which is only the
#: builtin ``TimeoutError`` from Python 3.11 on; catch both.
TIMEOUTS = (TimeoutError, asyncio.TimeoutError)

#: Properties never returned to a caller: vectors and bulky bookkeeping.
_BULKY_PROPS = frozenset({"source_chunk_ids"})
#: Keys of the SDK's ``__GraphRAGConfig__`` node (never shown).
_CONFIG_KEYS = frozenset({"sdk_version", "embedding_dimension"})
#: A list of at least this many numbers is treated as a vector and omitted.
VECTOR_MIN_LENGTH = 64
#: Longest string returned in one cell.
MAX_CELL_CHARS = 2000

TRAVERSE_DEFAULT_DEPTH, TRAVERSE_MAX_DEPTH = 2, 5
TRAVERSE_DEFAULT_EXPANSIONS, TRAVERSE_MAX_EXPANSIONS = 50, 200
TRAVERSE_DEFAULT_TIMEOUT_MS, TRAVERSE_MIN_TIMEOUT_MS, TRAVERSE_MAX_TIMEOUT_MS = 1000, 50, 5000
TRAVERSE_MAX_VIA = 5
TRAVERSE_PATH_LIMIT = 200

LOOKUP_DEFAULT_LIMIT, LOOKUP_MAX_LIMIT = 5, 25
LOOKUP_TIMEOUT_MS = 1000
_CANDIDATE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_CANDIDATE_STOP = frozenset(
    {"the", "and", "for", "with", "from", "into", "that", "this", "what", "which", "entity"}
)


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Coerce ``value`` to an int within ``[low, high]``; ``default`` if unusable."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


class _NoGraph:
    """The result of reading a graph that does not exist yet: no rows."""

    header: list[Any] = []
    result_set: list[Any] = []


def _is_missing_graph(exc: Exception) -> bool:
    # FalkorDB answers any query on a graph key that was never written with
    # "Invalid graph operation on empty key"; for a read that is just no rows.
    return "empty key" in str(exc)


def _is_server_timeout(exc: BaseException) -> bool:
    # FalkorDB's own TIMEOUT answers "Query timed out"; the connection wraps
    # it in a DatabaseError, which the TIMEOUTS handlers would not catch.
    return "timed out" in str(exc).lower()


async def bounded_query(
    store: Any,
    cypher: str,
    params: dict[str, Any] | None,
    timeout_ms: int,
) -> Any:
    """Run a read-only query with a database timeout and a client-side one."""
    try:
        return await asyncio.wait_for(
            store.query_raw(cypher, params or {}, read_only=True, timeout=timeout_ms),
            timeout=timeout_ms / 1000 + 0.1,
        )
    except TIMEOUTS:
        raise
    except Exception as exc:
        if _is_missing_graph(exc):
            return _NoGraph()
        if _is_server_timeout(exc):
            raise TimeoutError(str(exc)) from exc
        raise


async def _read(store: Any, cypher: str, params: dict[str, Any] | None = None) -> Any:
    try:
        return await store.query_raw(cypher, params or {}, read_only=True)
    except Exception as exc:
        if _is_missing_graph(exc):
            return _NoGraph()
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


def _public_labels(labels: Any) -> list[str]:
    return [str(lbl) for lbl in labels or [] if not str(lbl).startswith("__")]


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
    if isinstance(value, dict):
        if _CONFIG_KEYS <= set(value):
            return "[internal node omitted]"
        return {
            str(k): to_jsonable(v, max_chars=max_chars)
            for k, v in value.items()
            if "embedding" not in str(k).lower() and k not in _BULKY_PROPS
        }
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        labels = getattr(value, "labels", None)
        if labels is not None and any(lbl in INTERNAL_NODE_DENYLIST for lbl in labels):
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


def column_names(result: Any) -> list[str]:
    header = list(getattr(result, "header", None) or [])
    return [str(h[1]) if isinstance(h, list | tuple) and len(h) > 1 else str(h) for h in header]


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
    per hop is enough.
    """
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    visited = set(seed_ids)
    frontier = list(seed_ids)
    for _ in range(max(1, hops)):
        if not frontier or len(triples) >= cap:
            break
        result = await _read(
            store,
            "MATCH (a:__Entity__)-[r:RELATES]-(b:__Entity__) "
            "WHERE a.id IN $ids "
            "RETURN r.src_name, r.rel_type, r.tgt_name, "
            "coalesce(r.fact, r.description, ''), b.id "
            "ORDER BY r.src_name, r.rel_type, r.tgt_name "
            "LIMIT $limit",
            {"ids": frontier, "limit": per_hop_limit},
        )
        next_frontier: list[str] = []
        for row in _rows(result):
            src, rel_type, tgt = row[0] or "", row[1] or "", row[2] or ""
            fact = row[3] if len(row) > 3 else ""
            other = row[4] if len(row) > 4 else ""
            key = (str(src).lower(), str(rel_type), str(tgt).lower())
            if src and rel_type and tgt and key not in seen:
                seen.add(key)
                triples.append(
                    {
                        "source": str(src),
                        "type": str(rel_type),
                        "target": str(tgt),
                        "fact": str(fact or ""),
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


async def entity_names(store: Any, ids: list[str]) -> dict[str, str]:
    """id -> name for the given entity ids."""
    if not ids:
        return {}
    result = await _read(
        store,
        "UNWIND $ids AS eid MATCH (e:__Entity__ {id: eid}) RETURN e.id, e.name",
        {"ids": list(dict.fromkeys(ids))},
    )
    return {str(r[0]): str(r[1] or r[0]) for r in _rows(result) if r and r[0] is not None}


# ── Entity lookup ────────────────────────────────────────────────


async def find_entities(
    store: Any,
    text: str,
    *,
    limit: int = LOOKUP_DEFAULT_LIMIT,
    timeout_ms: int = LOOKUP_TIMEOUT_MS,
) -> list[dict[str, str]] | None:
    """Entities whose name contains ``text`` or its words, best match first.

    An exact (case-insensitive) name ranks first, then a name containing the
    whole text, then names sharing more of its words, then shorter names.
    Returns ``None`` when the lookup itself failed (so a caller does not word
    that as "no such entity").
    """
    needle = " ".join(str(text or "").lower().split())
    if not needle:
        return []
    tokens = [
        t
        for t in dict.fromkeys(_CANDIDATE_TOKEN_RE.findall(needle))
        if len(t) >= 3 and t not in _CANDIDATE_STOP
    ][:6]
    cypher = (
        "MATCH (e:__Entity__) WHERE e.name IS NOT NULL "
        "WITH e, toLower(toString(e.name)) AS n "
        "WITH e, n, size([t IN $tokens WHERE n CONTAINS t]) AS hits, "
        "n CONTAINS $needle AS whole, n = $needle AS exact "
        "WHERE whole OR hits > 0 "
        "RETURN e.id, e.name, [l IN labels(e) WHERE NOT l STARTS WITH '__'][0], e.description "
        "ORDER BY exact DESC, whole DESC, hits DESC, size(n) ASC, e.name LIMIT $limit"
    )
    try:
        result = await bounded_query(
            store, cypher, {"needle": needle, "tokens": tokens, "limit": limit}, timeout_ms
        )
    except LatencyBudgetExceededError:
        raise
    except Exception as exc:
        logger.warning("Entity lookup failed: %s", exc)
        return None
    found = []
    for row in _rows(result)[:limit]:
        if len(row) < 2 or row[1] is None:
            continue
        found.append(
            {
                "id": str(row[0]),
                "name": str(row[1]),
                "label": str(row[2]) if len(row) > 2 and row[2] else "",
                "description": str(row[3]) if len(row) > 3 and row[3] else "",
            }
        )
    return found


def render_candidates(candidates: list[dict[str, str]]) -> str:
    def one(c: dict[str, str]) -> str:
        name = c["name"] if len(c["name"]) <= 120 else c["name"][:120] + "…"
        label = f"{c['label']}, " if c.get("label") else ""
        return f'"{name}" ({label}id {c["id"]})'

    return "; ".join(one(c) for c in candidates)


async def resolve_start(store: Any, start: str, deadline: float) -> tuple[str, str]:
    """Return ``(id, name)`` of the one entity ``start`` names, or raise ToolError.

    Exact id or name first, then a case-insensitive name. An ambiguous or
    unknown name raises with the candidates to retry with. Every lookup
    draws on the caller's time budget (``deadline``).
    """

    def left_ms() -> int:
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ToolError(
                "Resolving the start entity used up the time budget.",
                hint="Retry with a larger timeout_ms or an exact entity id.",
            )
        return remaining

    if not start:
        raise ToolError("'start' is empty.", hint="Pass an entity name or id from lookup_entity.")
    for cypher in (
        "MATCH (e:__Entity__) WHERE e.id = $k OR e.name = $k RETURN e.id, e.name LIMIT 6",
        "MATCH (e:__Entity__) WHERE toLower(toString(e.name)) = toLower($k) "
        "RETURN e.id, e.name LIMIT 6",
    ):
        try:
            result = await bounded_query(store, cypher, {"k": start}, left_ms())
        except TIMEOUTS as exc:
            raise ToolError(
                "Resolving the start entity timed out.",
                hint="Retry with an exact entity id from lookup_entity.",
            ) from exc
        rows = [r for r in _rows(result) if r and r[0] is not None]
        if len(rows) == 1:
            return str(rows[0][0]), str(rows[0][1])
        if len(rows) > 1:
            listed = "; ".join(f'"{r[1]}" (id {r[0]})' for r in rows[:5])
            raise ToolError(
                f"{start!r} matches {len(rows)} entities: {listed}.",
                hint="Retry with one of these ids.",
            )
    candidates = await find_entities(store, start, timeout_ms=min(LOOKUP_TIMEOUT_MS, left_ms()))
    if candidates:
        raise ToolError(
            f"No entity is called {start!r}. Did you mean: {render_candidates(candidates)}?",
            hint="Retry with one of these exact names or ids.",
        )
    raise ToolError(
        f"No entity is called {start!r}.",
        hint="Call lookup_entity or retrieve first to find the exact name.",
    )


# ── Traverse ─────────────────────────────────────────────────────


async def expand(
    store: Any,
    start_id: str,
    start_name: str,
    depth_cap: int,
    expansions_cap: int,
    remaining_ms: Callable[[], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Breadth-first walk from ``start_id``; one query per hop.

    Returns ``(nodes, edges, stopped)``: the entities reached (with depth and
    the edge that reached them), every distinct edge seen with its fact, and
    which budget stopped the walk (``"complete"`` when none did). The
    remaining edge allowance goes into each hop's ``LIMIT``.
    """
    cypher = (
        "MATCH (s:__Entity__)-[r]-(t:__Entity__) WHERE s.id IN $frontier "
        "AND NOT type(r) IN $structural "
        "RETURN s.id, s.name, t.id, t.name, coalesce(r.rel_type, type(r)), "
        "ID(startNode(r)) = ID(s), coalesce(r.fact, r.description, '') "
        "ORDER BY s.id, t.id LIMIT $remaining"
    )
    frontier = [start_id]
    visited = {start_id}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    expansions = 0
    depth = 0
    stopped = "complete"
    while frontier and depth < depth_cap:
        left_ms = remaining_ms()
        if left_ms <= 0:
            stopped = "time_budget"
            break
        allowance = expansions_cap - expansions
        if allowance <= 0:
            stopped = "expansion_budget"
            break
        params = {
            "frontier": frontier,
            "remaining": allowance,
            "structural": sorted(STRUCTURAL_RELATIONSHIPS),
        }
        try:
            result = await bounded_query(store, cypher, params, left_ms)
        except TIMEOUTS:
            stopped = "time_budget"
            break
        depth += 1
        next_frontier = []
        for row in _rows(result):
            expansions += 1
            sid, sname, tid, tname, relation = row[0], row[1], row[2], row[3], row[4]
            outgoing = bool(row[5]) if len(row) > 5 else True
            fact = str(row[6] or "") if len(row) > 6 else ""
            if tid is None:
                continue
            src, tgt = ((sid, sname), (tid, tname)) if outgoing else ((tid, tname), (sid, sname))
            key = (str(src[0]), str(relation), str(tgt[0]), fact)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(
                    {
                        "source_id": str(src[0]),
                        "source": str(src[1] or src[0]),
                        "relation": str(relation),
                        "target_id": str(tgt[0]),
                        "target": str(tgt[1] or tgt[0]),
                        "fact": fact,
                    }
                )
            if tid in visited:
                continue
            visited.add(tid)
            next_frontier.append(tid)
            nodes.append(
                {
                    "id": str(tid),
                    "name": str(tname or tid),
                    "depth": depth,
                    "from_id": str(sid),
                    "relation": str(relation),
                    "outgoing": outgoing,
                }
            )
        frontier = next_frontier
        if expansions >= expansions_cap:
            stopped = "expansion_budget"
            break
    if stopped == "complete" and frontier and depth >= depth_cap:
        stopped = "depth_budget"
    return nodes, edges, stopped


async def typed_path(
    store: Any,
    start_id: str,
    via: list[str],
    timeout_ms: int,
) -> tuple[list[dict[str, Any]], str]:
    """Entities at the end of ``start -via[0]-> ... -via[-1]->`` (either direction per hop).

    Returns ``(nodes, stopped)``; ``stopped`` is ``"return_limit"`` when more
    than :data:`TRAVERSE_PATH_LIMIT` entities matched.
    """
    # The pattern is built from the hop count only; relation names are parameters.
    pattern = "".join(f"-[r{i}:RELATES]-(n{i}:__Entity__)" for i in range(len(via)))
    conditions = " AND ".join(f"r{i}.rel_type = $t{i}" for i in range(len(via)))
    last = f"n{len(via) - 1}"
    cypher = (
        f"MATCH (s:__Entity__){pattern} WHERE s.id = $start AND {conditions} "
        f"RETURN DISTINCT {last}.id, {last}.name LIMIT {TRAVERSE_PATH_LIMIT + 1}"
    )
    params: dict[str, Any] = {"start": start_id, **{f"t{i}": t for i, t in enumerate(via)}}
    try:
        result = await bounded_query(store, cypher, params, timeout_ms)
    except TIMEOUTS:
        return [], "time_budget"
    rows = [r for r in _rows(result) if r and r[0] is not None]
    truncated = len(rows) > TRAVERSE_PATH_LIMIT
    nodes = [
        {"id": str(r[0]), "name": str(r[1] or r[0]), "depth": len(via), "relation": via[-1]}
        for r in rows[:TRAVERSE_PATH_LIMIT]
    ]
    return nodes, "return_limit" if truncated else "complete"
