# GraphRAG SDK — Agentic Retrieval: graph tools
# The default toolset: semantic search, a natural-language graph query
# (text-to-Cypher under the read-only guard), a bounded graph walk and an
# entity lookup. Every result the model may rely on is numbered in the run's
# evidence ledger, so answers can cite it as [N].

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import Ontology, RetrieverResultItem
from graphrag_sdk.retrieval.agentic.cypher_guard import (
    STRUCTURAL_RELATIONSHIPS,
    GraphSchema,
    enforce_row_cap,
    introspect_schema,
    mask_internal_labels,
    validate_read_query,
)
from graphrag_sdk.retrieval.agentic.tools import (
    Tool,
    ToolContext,
    ToolRefusal,
    ToolRegistry,
    ToolResult,
    make_skill_tool,
    object_schema,
)

logger = logging.getLogger(__name__)

OntologyGetter = Callable[[], "Ontology | None"]

# ── Defaults (the hard ceilings live in the clamps below) ────────

QUERY_MAX_ROWS = 25
QUERY_MAX_CELL_CHARS = 200
QUERY_TIMEOUT_MS = 5000
QUERY_ATTEMPTS = 3

TRAVERSE_DEFAULT_DEPTH, TRAVERSE_MAX_DEPTH = 2, 5
TRAVERSE_DEFAULT_EXPANSIONS, TRAVERSE_MAX_EXPANSIONS = 50, 200
TRAVERSE_DEFAULT_TIMEOUT_MS, TRAVERSE_MIN_TIMEOUT_MS, TRAVERSE_MAX_TIMEOUT_MS = 1000, 50, 5000
TRAVERSE_MAX_VIA = 5
TRAVERSE_PATH_LIMIT = 200

LOOKUP_MAX_CANDIDATES = 5
LOOKUP_TIMEOUT_MS = 1000
_CANDIDATE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_CANDIDATE_STOP = frozenset(
    {"the", "and", "for", "with", "from", "into", "that", "this", "what", "which", "entity"}
)

_SEARCH_LINE_SECTIONS = {
    "entities": "entity",
    "relationships": "relationship",
    "facts": "fact",
    "cypher_results": "graph_query",
}
_SOURCE_PREFIX_RE = re.compile(r"^\[Source:\s*(.*?)\]\s*\n", re.DOTALL)

SEARCH_OVERRIDES = (
    "chunk_top_k",
    "max_entities",
    "max_relationships",
    "rel_top_k",
    "max_cypher_out",
    "max_entities_out",
    "max_relationships_out",
    "max_facts_out",
    "max_passages_out",
)


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    """Coerce ``value`` to an int within ``[low, high]``; ``default`` if unusable."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


async def _bounded_query(
    graph_store: Any,
    cypher: str,
    params: dict[str, Any] | None,
    timeout_ms: int,
) -> Any:
    """Run a read-only query with a database timeout and a client-side one."""
    return await asyncio.wait_for(
        graph_store.query_raw(cypher, params or {}, read_only=True, timeout=timeout_ms),
        timeout=timeout_ms / 1000 + 0.5,
    )


def format_value(value: Any) -> Any:
    """Render a Cypher result value readably for the LLM.

    FalkorDB returns Node/Edge objects whose ``str()`` is opaque. Surface
    their properties instead, without embedding vectors.
    """
    props = getattr(value, "properties", None)
    if isinstance(props, dict):
        labels = getattr(value, "labels", None) or getattr(value, "relation", None)
        readable = {k: v for k, v in props.items() if "embedding" not in k.lower()}
        if labels:
            return {"labels": labels, **readable}
        return readable
    if isinstance(value, dict):
        return {k: format_value(v) for k, v in value.items() if "embedding" not in str(k).lower()}
    if isinstance(value, list | tuple):
        return [format_value(v) for v in value]
    return value


def _cell(value: Any, max_chars: int) -> str:
    text = mask_internal_labels(str(format_value(value)))
    return text if len(text) <= max_chars else text[:max_chars] + "…[truncated]"


# ── search ───────────────────────────────────────────────────────


def split_search_items(items: list[RetrieverResultItem]) -> list[tuple[str, str, str]]:
    """Split retrieval sections into citable ``(kind, text, source)`` entries.

    MultiPath returns one item per section ("## Key Entities\\n- a\\n- b",
    passages joined by ``---``). Each line / passage becomes its own entry so
    a citation points at the statement it supports, not a whole section.
    """
    entries: list[tuple[str, str, str]] = []
    for item in items:
        section = str((item.metadata or {}).get("section", ""))
        content = (item.content or "").strip()
        if not content or section == "hint":
            continue
        if section in _SEARCH_LINE_SECTIONS:
            kind = _SEARCH_LINE_SECTIONS[section]
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("- ") and line[2:].strip():
                    entries.append((kind, line[2:].strip(), ""))
            continue
        if section == "passages":
            body = content.split("\n", 1)[1] if content.startswith("## ") else content
            for passage in body.split("\n---\n"):
                passage = passage.strip()
                if not passage:
                    continue
                source = ""
                match = _SOURCE_PREFIX_RE.match(passage)
                if match:
                    source = match.group(1).strip()
                    passage = passage[match.end() :].strip()
                if passage:
                    entries.append(("passage", passage, source))
            continue
        entries.append(("context", content, str((item.metadata or {}).get("source", ""))))
    return entries


def make_search_tool(strategy: Any, *, max_items: int = 40) -> Tool:
    """Semantic search; each returned statement or passage is numbered."""

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> ToolResult:
        query = str(tool_input["query"]).strip()
        overrides = {k: tool_input[k] for k in SEARCH_OVERRIDES if k in tool_input}
        result = await strategy.search(query, tctx.ctx, **overrides)
        entries = split_search_items(list(result.items))[:max_items]
        if not entries:
            return ToolResult("No results. Try different keywords or a narrower query.")
        lines = []
        numbers = []
        for kind, text, source in entries:
            ev = tctx.evidence.add(tool="search", kind=kind, content=text, source=source)
            numbers.append(ev.n)
            prefix = f"(source: {source}) " if source else ""
            lines.append(f"[{ev.n}] {prefix}{text}")
        return ToolResult("\n".join(lines), data={"evidence": numbers})

    properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "Self-contained search query with concrete names, not pronouns.",
        }
    }
    for key in SEARCH_OVERRIDES:
        properties[key] = {"type": "integer", "description": "Optional retrieval limit."}
    return Tool(
        name="search",
        description=(
            "Semantic search of the knowledge graph. Returns numbered entities, "
            "relationships, facts and source passages relevant to the query. Use it "
            "first for any question about the content."
        ),
        handler=handler,
        parameters=object_schema(properties, ["query"]),
    )


# ── query_graph (text-to-Cypher) ─────────────────────────────────


def build_query_prompt(
    question: str,
    schema: GraphSchema,
    ontology: Ontology | None,
    *,
    max_rows: int,
    previous_error: str = "",
) -> str:
    from graphrag_sdk.retrieval.strategies.cypher_generation import render_ontology_block

    relations = ", ".join(schema.semantic_relations) or "(none declared)"
    prompt = (
        "You write ONE read-only Cypher query for a FalkorDB knowledge graph.\n\n"
        "## Ontology\n"
        f"{render_ontology_block(ontology)}\n\n"
        "## Live graph schema\n"
        f"Node labels: {', '.join(schema.labels) or '(unknown)'}\n"
        f"Relationship types: {', '.join(schema.relationship_types) or '(unknown)'}\n"
        f"Property keys: {', '.join(schema.property_keys) or '(unknown)'}\n\n"
        "## Rules\n"
        "- Start with MATCH, OPTIONAL MATCH, UNWIND or WITH. Never CREATE, MERGE, "
        "SET, DELETE, REMOVE, DROP, FOREACH, CALL or LOAD CSV.\n"
        "- Use only the labels, relationship types and properties listed above.\n"
        "- Relationships between entities are stored as [r:RELATES] with the relation "
        f"name in r.rel_type (one of: {relations}). Give every RELATES edge its own "
        "variable and constrain its rel_type; never write the relation name inside "
        "the brackets.\n"
        "- Match entities by name or id, return scalar values (e.name, count(...)) "
        "with unique aliases, never whole nodes or embedding properties.\n"
        "- Prefer aggregation (count, collect, ORDER BY ... LIMIT) for counting, "
        f"ranking and grouping questions. At most {max_rows} rows come back.\n"
    )
    if previous_error:
        prompt += f"\nThe previous attempt was rejected: {previous_error}\nFix it.\n"
    return prompt + f"\n## Question\n{question}\n\nReturn ONLY the query in a ```cypher block."


def _column_names(result: Any) -> list[str]:
    header = list(getattr(result, "header", None) or [])
    return [str(h[1]) if isinstance(h, list | tuple) and len(h) > 1 else str(h) for h in header]


def make_query_graph_tool(
    graph_store: Any,
    llm: Any,
    *,
    ontology_getter: OntologyGetter | None = None,
    max_rows: int = QUERY_MAX_ROWS,
    max_cell_chars: int = QUERY_MAX_CELL_CHARS,
    timeout_ms: int = QUERY_TIMEOUT_MS,
    attempts: int = QUERY_ATTEMPTS,
) -> Tool:
    """Answer a counting / ranking / connection question with a graph query.

    The question is turned into Cypher by the LLM, checked by
    :func:`validate_read_query`, row-capped, and run read-only with a timeout.
    A rejected or failing query is retried with the reason (``attempts`` in
    all). Every query tried is recorded in ``tctx.state["generated_cypher"]``.
    """
    from graphrag_sdk.retrieval.strategies.cypher_generation import extract_cypher

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> ToolResult:
        question = str(tool_input["question"]).strip()
        ctx = tctx.ctx
        ontology = ontology_getter() if ontology_getter else None
        schema = tctx.state.get("graph_schema")
        if schema is None:
            schema = await introspect_schema(graph_store, ontology)
            tctx.state["graph_schema"] = schema
        log: list[dict[str, Any]] = tctx.state.setdefault("generated_cypher", [])

        last_error = ""
        last_cypher = ""
        for _attempt in range(attempts):
            ctx.ensure_budget("agent query_graph generation")
            prompt = build_query_prompt(
                question, schema, ontology, max_rows=max_rows, previous_error=last_error
            )
            response = await llm.ainvoke(
                prompt, timeout=ctx.provider_timeout_seconds("agent query_graph generation")
            )
            cypher = extract_cypher(response.content or "")
            if not cypher:
                last_error = "no Cypher query was produced"
                continue
            last_cypher = cypher
            problems = validate_read_query(cypher, schema)
            if problems:
                last_error = "; ".join(problems)
                continue
            cypher = enforce_row_cap(cypher, max_rows)
            started = time.monotonic()
            try:
                result = await _bounded_query(graph_store, cypher, None, timeout_ms)
            except LatencyBudgetExceededError:
                raise
            except TimeoutError:
                last_error = f"the query took longer than {timeout_ms} ms"
                continue
            except Exception as exc:
                last_error = f"the database rejected the query: {str(exc)[:200]}"
                continue
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            rows = list(getattr(result, "result_set", None) or [])[:max_rows]
            columns = _column_names(result)
            log.append(
                {"question": question, "query": cypher, "rows": len(rows), "elapsed_ms": elapsed_ms}
            )
            rendered = (
                "\n".join(" | ".join(_cell(c, max_cell_chars) for c in row) for row in rows)
                or "(no rows)"
            )
            content = (
                f"Graph query result for: {question}\n"
                f"Cypher: {mask_internal_labels(cypher)}\n"
                f"Columns: {', '.join(columns)}\nRows:\n{rendered}"
            )
            ev = tctx.evidence.add(
                tool="query_graph", kind="graph_query", content=content, source="graph query"
            )
            return ToolResult(
                f"[{ev.n}] {content}",
                data={"evidence": [ev.n], "cypher": cypher, "rows": len(rows)},
            )

        if last_cypher:
            log.append({"question": question, "query": last_cypher, "rows": 0, "error": last_error})
        raise ToolRefusal(
            f"no valid graph query could be built ({last_error}). Answer from the search "
            "results you already have, or rephrase the question."
        )

    return Tool(
        name="query_graph",
        description=(
            "Answer a question with a read-only graph query: counts, rankings, "
            "aggregates, exact enumerations, and questions whose answer is a "
            "connection between entities ('which X are linked to Y'). Takes the "
            f"question in plain language; returns at most {max_rows} rows as one "
            "numbered result. Ask for a count or a top-N rather than a full listing."
        ),
        handler=handler,
        parameters=object_schema(
            {"question": {"type": "string", "description": "The question in plain language."}},
            ["question"],
        ),
    )


# ── entity lookup ────────────────────────────────────────────────


async def find_entities(
    graph_store: Any,
    text: str,
    *,
    limit: int = LOOKUP_MAX_CANDIDATES,
    timeout_ms: int = LOOKUP_TIMEOUT_MS,
) -> list[dict[str, str]] | None:
    """Entities whose name contains ``text`` or its words, best match first.

    A name containing the whole text ranks first, then names sharing more of
    its words, then shorter names. Returns ``None`` when the lookup itself
    failed (so a caller does not word that as "no such entity").
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
        "WITH e, n, size([t IN $tokens WHERE n CONTAINS t]) AS hits, n CONTAINS $needle AS whole "
        "WHERE whole OR hits > 0 "
        "RETURN e.id, e.name, [l IN labels(e) WHERE l <> '__Entity__'][0], e.description "
        "ORDER BY whole DESC, hits DESC, size(n) ASC LIMIT $limit"
    )
    try:
        result = await _bounded_query(
            graph_store, cypher, {"needle": needle, "tokens": tokens, "limit": limit}, timeout_ms
        )
    except LatencyBudgetExceededError:
        raise
    except Exception as exc:
        logger.warning("Entity lookup failed: %s", exc)
        return None
    found = []
    for row in (getattr(result, "result_set", None) or [])[:limit]:
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


def _render_candidates(candidates: list[dict[str, str]]) -> str:
    def one(c: dict[str, str]) -> str:
        name = c["name"] if len(c["name"]) <= 120 else c["name"][:120] + "…"
        label = f"{c['label']}, " if c.get("label") else ""
        return f'"{name}" ({label}id {c["id"]})'

    return "; ".join(one(c) for c in candidates)


def make_lookup_entity_tool(graph_store: Any) -> Tool:
    """Find the exact name and id of entities matching a partial name."""

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> ToolResult:
        name = str(tool_input["name"]).strip()
        found = await find_entities(graph_store, name)
        if found is None:
            raise ToolRefusal("the entity lookup is unavailable right now")
        if not found:
            return ToolResult(f"No entity name matches {name!r}.")
        lines = []
        for c in found:
            desc = f": {c['description'][:160]}" if c["description"] else ""
            label = f"{c['label']}, " if c["label"] else ""
            lines.append(f'- "{c["name"]}" ({label}id {c["id"]}){desc}')
        return ToolResult("Matching entities:\n" + "\n".join(lines), data={"entities": found})

    return Tool(
        name="lookup_entity",
        description=(
            "Find entities by (part of) their name. Returns up to 5 exact names and ids "
            "to use with traverse. Not a source to cite."
        ),
        handler=handler,
        parameters=object_schema(
            {"name": {"type": "string", "description": "A name or part of one."}}, ["name"]
        ),
    )


# ── traverse (bounded walk) ──────────────────────────────────────


async def _resolve_start(graph_store: Any, start: str, timeout_ms: int) -> tuple[str, str]:
    """Return ``(id, name)`` of the one entity ``start`` names, or refuse."""
    for cypher in (
        "MATCH (e:__Entity__) WHERE e.id = $k OR e.name = $k RETURN e.id, e.name LIMIT 6",
        "MATCH (e:__Entity__) WHERE toLower(toString(e.name)) = toLower($k) "
        "RETURN e.id, e.name LIMIT 6",
    ):
        try:
            result = await _bounded_query(graph_store, cypher, {"k": start}, timeout_ms)
        except TimeoutError as exc:
            raise ToolRefusal("resolving the start entity timed out") from exc
        rows = [r for r in (getattr(result, "result_set", None) or []) if r and r[0] is not None]
        if len(rows) == 1:
            return str(rows[0][0]), str(rows[0][1])
        if len(rows) > 1:
            listed = "; ".join(f'"{r[1]}" (id {r[0]})' for r in rows[:5])
            raise ToolRefusal(
                f"{start!r} matches {len(rows)} entities ({listed}). Retry with one of these ids."
            )
    candidates = await find_entities(graph_store, start)
    if candidates:
        raise ToolRefusal(
            f"no entity is called {start!r}. Did you mean: {_render_candidates(candidates)}? "
            "Retry with one of these exact names or ids."
        )
    if candidates is None:
        raise ToolRefusal(f"no entity is called {start!r} (suggestions unavailable)")
    raise ToolRefusal(f"no entity is called {start!r}; use lookup_entity or search first")


def make_traverse_tool(
    graph_store: Any,
    *,
    max_depth: int = TRAVERSE_MAX_DEPTH,
    max_expansions: int = TRAVERSE_MAX_EXPANSIONS,
    max_timeout_ms: int = TRAVERSE_MAX_TIMEOUT_MS,
) -> Tool:
    """Bounded walk from one entity: breadth-first, or along typed relations.

    Without ``via`` the walk expands breadth-first, one query per hop, with
    the remaining edge allowance pushed into each hop's ``LIMIT``. With
    ``via`` it follows one relation per hop and returns only where the path
    ends. Every result says which budget, if any, stopped it.
    """

    async def handler(tool_input: dict[str, Any], tctx: ToolContext) -> ToolResult:
        started = time.monotonic()
        depth_cap = clamp_int(tool_input.get("max_depth"), TRAVERSE_DEFAULT_DEPTH, 1, max_depth)
        expansions_cap = clamp_int(
            tool_input.get("max_expansions"), TRAVERSE_DEFAULT_EXPANSIONS, 1, max_expansions
        )
        budget_ms = clamp_int(
            tool_input.get("timeout_ms"),
            min(TRAVERSE_DEFAULT_TIMEOUT_MS, max_timeout_ms),
            TRAVERSE_MIN_TIMEOUT_MS,
            max_timeout_ms,
        )
        deadline = started + budget_ms / 1000
        via_raw = tool_input.get("via") or []
        via = [str(v).strip() for v in via_raw if str(v).strip()]
        if len(via) > TRAVERSE_MAX_VIA:
            raise ToolRefusal(f"'via' may name at most {TRAVERSE_MAX_VIA} relations")

        start_id, start_name = await _resolve_start(
            graph_store, str(tool_input["start"]).strip(), budget_ms
        )

        def remaining_ms() -> int:
            return int((deadline - time.monotonic()) * 1000)

        if via:
            text, nodes, stopped = await _typed_path(
                graph_store, start_id, via, max(remaining_ms(), 1)
            )
        else:
            text, nodes, stopped = await _expand(
                graph_store, start_id, depth_cap, expansions_cap, remaining_ms
            )
        header = (
            f"Walk from {start_name!r} (id {start_id})"
            + (f" along {' -> '.join(via)}" if via else f", depth <= {depth_cap}")
            + f"; reached {len(nodes)} entities; stopped: {stopped}."
        )
        content = header + ("\n" + text if text else "")
        ev = tctx.evidence.add(
            tool="traverse", kind="traversal", content=content, source="graph traversal"
        )
        return ToolResult(
            f"[{ev.n}] {content}",
            data={"evidence": [ev.n], "start_id": start_id, "nodes": nodes, "stopped": stopped},
        )

    return Tool(
        name="traverse",
        description=(
            "Walk the graph from one entity (exact name or id) under explicit bounds. "
            "Without 'via' it lists the neighbourhood breadth-first; with 'via' "
            "(relation names, one per hop) it follows that path and returns where it "
            "ends. A refused call lists the exact names to retry with."
        ),
        handler=handler,
        parameters=object_schema(
            {
                "start": {"type": "string", "description": "Exact entity name or id."},
                "via": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Relation names to follow in order (max {TRAVERSE_MAX_VIA}).",
                },
                "max_depth": {"type": "integer", "description": f"1-{max_depth}, default 2."},
                "max_expansions": {
                    "type": "integer",
                    "description": f"Edges examined, 1-{max_expansions}, default 50.",
                },
                "timeout_ms": {"type": "integer", "description": "Time budget in ms."},
            },
            ["start"],
        ),
    )


async def _expand(
    graph_store: Any,
    start_id: str,
    depth_cap: int,
    expansions_cap: int,
    remaining_ms: Callable[[], int],
) -> tuple[str, list[dict[str, Any]], str]:
    cypher = (
        "MATCH (s:__Entity__)-[r]-(t:__Entity__) WHERE s.id IN $frontier "
        "AND NOT type(r) IN $structural "
        "RETURN s.id, t.id, t.name, coalesce(r.rel_type, type(r)), "
        "ID(startNode(r)) = ID(s) ORDER BY s.id, t.id LIMIT $remaining"
    )
    frontier = [start_id]
    visited = {start_id}
    nodes: list[dict[str, Any]] = []
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
            result = await _bounded_query(graph_store, cypher, params, left_ms)
        except TimeoutError:
            stopped = "time_budget"
            break
        depth += 1
        next_frontier = []
        for row in getattr(result, "result_set", None) or []:
            expansions += 1
            source, target, name, predicate = row[0], row[1], row[2], row[3]
            outgoing = bool(row[4]) if len(row) > 4 else True
            if target is None or target in visited:
                continue
            visited.add(target)
            next_frontier.append(target)
            nodes.append(
                {
                    "id": str(target),
                    "name": name,
                    "depth": depth,
                    "from": str(source),
                    "relation": predicate,
                    "outgoing": outgoing,
                }
            )
        frontier = next_frontier
        if expansions >= expansions_cap:
            stopped = "expansion_budget"
            break
    if stopped == "complete" and frontier and depth >= depth_cap:
        stopped = "depth_budget"
    lines = [
        f"- {n['name']} (id {n['id']}) {'<-' if not n['outgoing'] else '->'} "
        f"{n['relation']} from {n['from']}, depth {n['depth']}"
        for n in nodes
    ]
    return "\n".join(lines), nodes, stopped


async def _typed_path(
    graph_store: Any,
    start_id: str,
    via: list[str],
    timeout_ms: int,
) -> tuple[str, list[dict[str, Any]], str]:
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
        result = await _bounded_query(graph_store, cypher, params, timeout_ms)
    except TimeoutError:
        return "", [], "time_budget"
    rows = [r for r in (getattr(result, "result_set", None) or []) if r and r[0] is not None]
    truncated = len(rows) > TRAVERSE_PATH_LIMIT
    nodes = [
        {"id": str(r[0]), "name": r[1], "depth": len(via), "relation": via[-1]}
        for r in rows[:TRAVERSE_PATH_LIMIT]
    ]
    lines = [f"- {n['name']} (id {n['id']})" for n in nodes]
    return "\n".join(lines), nodes, "return_limit" if truncated else "complete"


# ── Default registry ─────────────────────────────────────────────


def build_default_registry(
    *,
    strategy: Any | None = None,
    graph_store: Any | None = None,
    llm: Any | None = None,
    include_skills: bool = True,
    ontology_getter: OntologyGetter | None = None,
) -> ToolRegistry:
    """Assemble the standard agent toolset from available primitives.

    ``search`` needs a retrieval strategy; ``query_graph`` needs a graph
    store and an LLM; ``traverse``, ``lookup_entity`` and the skills need a
    graph store.
    """
    registry = ToolRegistry()
    if strategy is not None:
        registry.register(make_search_tool(strategy))
    if graph_store is not None:
        if llm is not None:
            registry.register(
                make_query_graph_tool(graph_store, llm, ontology_getter=ontology_getter)
            )
        registry.register(make_traverse_tool(graph_store))
        registry.register(make_lookup_entity_tool(graph_store))
        if include_skills:
            from graphrag_sdk.skills import SKILL_REGISTRY

            for skill_cls in SKILL_REGISTRY.values():
                registry.register(make_skill_tool(skill_cls(graph_store, llm)))
    return registry
