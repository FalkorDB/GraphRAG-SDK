# GraphRAG SDK — Retrieval: Text-to-Cypher Generation
# Adapted from FalkorDB/GraphRAG-SDK upstream for the unified RELATES schema.
# Generates read-only Cypher queries from natural language questions.
#
# Design: Uses typed entity node labels (Person, Organization, Location, etc.)
# for routing and aggregation rather than matching rel_type on RELATES edges.
# Cypher results go directly to the LLM context — NOT through the reranker.

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Valid labels for our graph schema ────────────────────────────

_ENTITY_LABELS = frozenset(
    {
        "Person",
        "Organization",
        "Technology",
        "Product",
        "Location",
        "Date",
        "Event",
        "Concept",
        "Law",
        "Dataset",
        "Method",
    }
)

_STRUCTURAL_LABELS = frozenset({"Chunk", "Document", "__Entity__"})

_ALL_LABELS = _ENTITY_LABELS | _STRUCTURAL_LABELS

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|CALL\s+db\.idx)\b",
    re.IGNORECASE,
)

# Default row cap auto-injected when the LLM's query lacks a LIMIT.
# Pure aggregations (count/sum/avg over no group-by) return one row, so
# they skip injection entirely; group-by lists need enough rows that a 10-org
# breakdown isn't truncated.
_DEFAULT_ROW_LIMIT = 100

_AGG_FN_NAMES = ("count", "sum", "avg", "min", "max", "collect",
                 "stdev", "percentileCont", "percentileDisc")
_AGG_FN_RE = re.compile(
    r"^\s*(?:" + "|".join(_AGG_FN_NAMES) + r")\s*\(",
    re.IGNORECASE,
)

# Detects FUNCTION-style calls under a dotted namespace, e.g.
# ``apoc.text.regexGroups(...)``, ``gds.shortest.path(...)``, ``db.idx.fulltext.queryNodes(...)``.
# FalkorDB does not implement APOC/GDS/db plugins, so any dotted-namespace
# function call silently returns 0 rows at execution. We reject these in the
# validator so the existing retry-with-feedback loop can correct the query.
# Note: ``\bCALL\b`` already catches procedure-style invocations; this regex
# specifically targets the function-style pattern that slips through.
_DOTTED_FN_RE = re.compile(
    r"\b([a-zA-Z_]\w*)\.([a-zA-Z_][\w.]*)\s*\(",
)

# Matches single- or double-quoted Cypher string literals, honoring backslash
# escapes so embedded quotes don't terminate the match early.
_STRING_LITERAL_RE = re.compile(
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"",
    re.DOTALL,
)


def _strip_string_literals(cypher: str) -> str:
    """Replace quoted string literals with empty quotes.

    Keyword/function blocklists (CALL, dotted-namespace fn calls, LOAD CSV,
    write keywords) must not match against text that lives inside a string
    predicate — e.g., ``WHERE n.text CONTAINS 'apoc.foo('``.
    """
    return _STRING_LITERAL_RE.sub("''", cypher)

# ── Schema prompt ────────────────────────────────────────────────

SCHEMA_PROMPT = """\
You are a Cypher query generator for a FalkorDB graph database.

## Graph Schema

### Entity node labels (all entities also carry the label `__Entity__`):
Person, Organization, Technology, Product, Location, Date, Event, Concept, Law, Dataset, Method

### Entity node properties:
- name (string) — entity name
- description (string) — entity description

### Edge types:
- RELATES: connects any entity to any entity.
  Properties: rel_type (string), fact (string — evidence text), src_name (string), tgt_name (string)
- MENTIONED_IN: connects entity to Chunk node (provenance)
- PART_OF: connects Document to Chunk
- NEXT_CHUNK: connects Chunk to next sequential Chunk

## FalkorDB-specific rules (CRITICAL — violating these causes execution errors):
1. Do NOT use shortestPath() or allShortestPaths() — FalkorDB returns
   Path objects that cause "Type mismatch: expected List or Null but was Path".
2. Every column in RETURN must have a UNIQUE name. Use aliases:
   `RETURN a.name AS a_name, b.name AS b_name` — NEVER return
   columns without aliases when both are `.name`.
3. Do NOT use the `path =` variable syntax. Instead use explicit node/edge variables.
4. Keep queries simple: 1-2 MATCH clauses maximum. Add LIMIT 25 to prevent huge result sets.
5. Use CONTAINS for fuzzy name matching: `WHERE e.name CONTAINS 'keyword'`
6. Generate READ-ONLY queries only — no CREATE, DELETE, SET, MERGE, REMOVE.
7. Always include a RETURN clause.

## Strategy: use entity TYPE LABELS for routing, not rel_type
Instead of guessing the exact rel_type string, leverage the typed entity labels:
- To find people related to something: `MATCH (p:Person)-[:RELATES]-(target)`
- To find locations: `MATCH (l:Location)-[:RELATES]-(e)`
- To find connections: `MATCH (a)-[:RELATES]-(b)` with entity name filters
- To count: `RETURN count(DISTINCT e)` or `RETURN count(r)`
- To list all of a type: `MATCH (e:Technology) RETURN e.name, e.description LIMIT 25`
- For "BOTH X AND Y" questions, use two MATCH clauses sharing the same
  variable to express set intersection — never UNION.

## Examples

Question: "Who is connected to the old lighthouse?"
```cypher
MATCH (e:__Entity__)-[r:RELATES]-(other:__Entity__)
WHERE e.name CONTAINS 'lighthouse'
RETURN other.name AS name, labels(other) AS type, r.rel_type AS relation, r.fact AS evidence
LIMIT 25
```

Question: "What locations are mentioned in the story?"
```cypher
MATCH (l:Location)
RETURN l.name AS location, l.description AS description
LIMIT 25
```

Question: "How are Alice and the castle connected?"
```cypher
MATCH (a:__Entity__)-[r1:RELATES]-(mid:__Entity__)-[r2:RELATES]-(b:__Entity__)
WHERE a.name CONTAINS 'Alice' AND b.name CONTAINS 'castle'
RETURN a.name AS from_entity, r1.rel_type AS rel1,
  mid.name AS via_entity, r2.rel_type AS rel2, b.name AS to_entity
LIMIT 15
```

Question: "How many people are in the story?"
```cypher
MATCH (p:Person)
RETURN count(p) AS person_count
```

Question: "What did the professor discover?"
```cypher
MATCH (p:Person)-[r:RELATES]->(thing:__Entity__)
WHERE p.name CONTAINS 'professor'
RETURN p.name AS person, thing.name AS discovery, r.rel_type AS relationship, r.fact AS evidence
LIMIT 20
```

Question: "Which city has the most employees mentioned?"
Note: a Person and a Location are typically NOT directly connected — they
share an organization. Use a 2-hop traversal through the intermediary so the
group-by works on the real graph topology:
```cypher
MATCH (p:Person)-[:RELATES]-(o:Organization)-[:RELATES]-(l:Location)
RETURN l.name AS city, count(DISTINCT p) AS employee_count
ORDER BY employee_count DESC
LIMIT 5
```

Question: "Who works at BOTH Acme and Globex?"
```cypher
MATCH (p:Person)-[:RELATES]-(o1:Organization),
      (p)-[:RELATES]-(o2:Organization)
WHERE o1.name CONTAINS 'Acme' AND o2.name CONTAINS 'Globex'
RETURN DISTINCT p.name AS person
```

## Your task

Generate a single Cypher query to answer the following question.
If you cannot generate a valid query, return an empty code block.
Return ONLY the Cypher query inside triple backticks.

Question: {question}
"""


# ── Cypher extraction ────────────────────────────────────────────


def extract_cypher(text: str) -> str:
    """Extract Cypher from LLM response, handling markdown code blocks."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    pattern = r"```(?:cypher)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    if text.upper().startswith(("MATCH", "OPTIONAL", "WITH", "CALL", "UNWIND")):
        return text.strip()
    return ""


# ── Cypher sanitization ─────────────────────────────────────────


def _split_top_level_commas(s: str) -> list[str]:
    """Split on commas that aren't inside parentheses/brackets/braces."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [p for p in out if p]


def _is_pure_aggregation(cypher: str) -> bool:
    """True iff the FINAL RETURN clause projects only aggregate functions.

    Pure aggregations (e.g., ``RETURN count(p)``) always return exactly one
    row, so auto-injecting LIMIT is a no-op. Group-by patterns
    (``RETURN o.name, count(p)``) are NOT pure — at least one projection is
    a non-aggregate dimension that the LIMIT would actually apply to.
    """
    # Find the final RETURN body, stopping at ORDER BY / SKIP / LIMIT / end.
    matches = list(re.finditer(
        r"\bRETURN\b\s+(.+?)(?=\bORDER\s+BY\b|\bSKIP\b|\bLIMIT\b|;|$)",
        cypher,
        re.IGNORECASE | re.DOTALL,
    ))
    if not matches:
        return False
    body = matches[-1].group(1).strip()
    if not body:
        return False
    projections = _split_top_level_commas(body)
    if not projections:
        return False
    return all(_AGG_FN_RE.match(p) for p in projections)


def _sanitize_cypher(cypher: str) -> str:
    """Fix common LLM-generated Cypher issues before execution.

    Catches patterns that would cause FalkorDB runtime errors.
    """
    # Remove shortestPath / allShortestPaths wrappers
    cypher = re.sub(
        r"\b(allShortestPaths|shortestPath)\s*\(",
        "(",
        cypher,
        flags=re.IGNORECASE,
    )
    # Remove path variable assignments: "path = MATCH" -> "MATCH"
    cypher = re.sub(r"\bpath\s*=\s*", "", cypher, flags=re.IGNORECASE)

    # Inject LIMIT only when the LLM didn't provide one AND the query isn't a
    # single-row pure aggregation. Pure aggregations don't benefit from a cap;
    # group-by lists do, but the previous default (25) was too small to fit a
    # full 10-org breakdown. Skip on aggregations to avoid a misleading no-op.
    if not re.search(r"\bLIMIT\b", cypher, re.IGNORECASE):
        if not _is_pure_aggregation(cypher):
            cypher = cypher.rstrip().rstrip(";") + f"\nLIMIT {_DEFAULT_ROW_LIMIT}"

    return cypher


# ── Cypher validation ────────────────────────────────────────────


def validate_cypher(cypher: str) -> list[str]:
    """Validate generated Cypher for safety and correctness.

    Uses an allowlist approach: the query must start with a read-only
    keyword, and dangerous constructs are explicitly rejected.

    Returns list of error strings; empty list means valid.
    """
    errors: list[str] = []
    if not cypher:
        errors.append("Empty Cypher query")
        return errors

    # Normalize: strip comments and trailing semicolons
    cypher_norm = re.sub(r"//.*?$", "", cypher, flags=re.MULTILINE)
    cypher_norm = re.sub(r"/\*.*?\*/", "", cypher_norm, flags=re.DOTALL)
    cypher_norm = re.sub(r";\s*$", "", cypher_norm).strip()

    if not cypher_norm:
        errors.append("Cypher query is empty after removing comments")
        return errors

    # Strip quoted string literals so keyword/function pattern checks don't
    # match against text inside predicates (e.g. ``WHERE n.text CONTAINS
    # 'apoc.foo('`` must not trip the APOC blocklist).
    cypher_code = _strip_string_literals(cypher_norm)

    # Allowlist: must start with a read-only keyword
    if not re.match(r"^(MATCH|OPTIONAL\s+MATCH|UNWIND|WITH)\b", cypher_norm, re.IGNORECASE):
        errors.append("Query must start with MATCH, OPTIONAL MATCH, UNWIND, or WITH")

    # Reject multi-statement queries
    if ";" in cypher_code:
        errors.append("Multiple Cypher statements are not allowed")

    # Reject procedures and bulk import
    if re.search(r"\bCALL\b", cypher_code, re.IGNORECASE):
        errors.append("CALL procedures are not allowed in generated queries")
    if re.search(r"\bLOAD\s+CSV\b", cypher_code, re.IGNORECASE):
        errors.append("LOAD CSV is not allowed in generated queries")

    # Reject dotted-namespace function calls (apoc.*, gds.*, db.*).
    # FalkorDB doesn't implement these plugins; the call silently returns 0
    # rows at execution. Surfacing it here lets the retry loop regenerate.
    for ns, _ in _DOTTED_FN_RE.findall(cypher_code):
        errors.append(
            f"Unsupported function namespace '{ns}.*' "
            "(FalkorDB does not implement APOC/GDS/db plugin functions). "
            "Use only built-in Cypher functions like count, sum, avg, "
            "labels, toInteger, substring, etc."
        )
        break  # one error is enough — the LLM only needs to fix the pattern

    # No write operations
    if _WRITE_KEYWORDS.search(cypher_code):
        errors.append("Write operation detected — query must be read-only")

    # Must have RETURN
    if not re.search(r"\bRETURN\b", cypher_norm, re.IGNORECASE):
        errors.append("Missing RETURN clause")

    # Check referenced labels exist in schema
    label_pattern = re.findall(r"\((?:\w+)?:(\w+)", cypher_norm)
    for label in label_pattern:
        if label not in _ALL_LABELS:
            errors.append(f"Unknown label: {label}")

    return errors


# ── Text-to-Cypher execution ────────────────────────────────────


async def generate_cypher(
    llm: Any,
    question: str,
    *,
    max_retries: int = 3,
) -> str | None:
    """Generate a Cypher query from a natural language question.

    Returns the Cypher string, or None if all retries fail.
    """
    prompt = SCHEMA_PROMPT.format(question=question)
    last_error = ""

    for attempt in range(max_retries):
        try:
            if attempt > 0 and last_error:
                prompt_with_feedback = (
                    prompt + f"\n\nPrevious attempt failed with error: {last_error}\n"
                    "Remember: no shortestPath, every RETURN column must have a "
                    "unique alias, add LIMIT, keep it simple."
                )
                response = await llm.ainvoke(prompt_with_feedback)
            else:
                response = await llm.ainvoke(prompt)

            cypher = extract_cypher(response.content)
            if not cypher:
                last_error = "Empty query generated"
                continue

            errors = validate_cypher(cypher)
            if errors:
                last_error = "; ".join(errors)
                continue

            # Sanitize before returning
            cypher = _sanitize_cypher(cypher)
            return cypher

        except Exception as exc:
            last_error = str(exc)
            logger.debug("Cypher generation attempt %d failed: %s", attempt + 1, exc)

    logger.debug("Cypher generation exhausted %d retries: %s", max_retries, last_error)
    return None


# Matches a typed entity label inside a node pattern so we can widen it to
# ``__Entity__`` on a 0-row retry. Captures the prefix (open paren + optional
# variable + colon) so ``re.sub`` keeps the surrounding shape intact.
_TYPED_NODE_LABEL_RE = re.compile(
    r"(\(\s*\w*\s*:)(" + "|".join(re.escape(l) for l in _ENTITY_LABELS) + r")\b"
)


def _widen_typed_labels(cypher: str) -> str:
    """Swap typed entity labels (``:Person`` etc.) inside node patterns to
    ``:__Entity__``. Used when a typed-label query returned 0 rows because
    the extractor labelled the entity differently than the LLM expected."""
    return _TYPED_NODE_LABEL_RE.sub(r"\1__Entity__", cypher)


def _should_widen_labels(cypher: str) -> bool:
    """Gate for the 0-row label-widen fallback.

    Skip widening when the typed label IS the filter — i.e. the RETURN
    aggregates over a labeled variable AND the query has no ``WHERE … CONTAINS``
    name predicate. In that case the user is asking "how many Persons?" and
    widening would change the semantics. Otherwise (typical case: typed label
    present alongside a name predicate or non-aggregate RETURN), widen.

    Conservative: when in doubt we skip the fallback rather than risk a
    semantically-different query.
    """
    if not _TYPED_NODE_LABEL_RE.search(cypher):
        return False  # nothing to widen
    has_contains_filter = bool(
        re.search(r"\bWHERE\b.*\bCONTAINS\b", cypher, re.IGNORECASE | re.DOTALL)
    )
    if has_contains_filter:
        return True
    # No name filter — check whether the RETURN is an aggregate over a labeled
    # variable. If so, widening turns "count Persons" into "count Entities".
    if _is_pure_aggregation(cypher):
        return False
    return True


def _parse_cypher_result_set(result_set: Any) -> tuple[list[str], dict[str, dict]]:
    """Turn a FalkorDB result_set into (fact_strings, entities)."""
    fact_strings: list[str] = []
    entities: dict[str, dict] = {}
    for row in result_set:
        parts: list[str] = []
        for val in row:
            if val is None:
                continue
            s = str(val).strip()
            if s and s != "[]" and s != "{}":
                parts.append(s)
        if not parts:
            continue
        fact_strings.append(" | ".join(parts))
        for val in row:
            if (
                isinstance(val, str)
                and len(val) > 1
                and len(val) < 100
                and not val.startswith(("(", "[", "{"))
                and not val.replace(".", "").isdigit()
            ):
                eid = val.strip().lower().replace(" ", "_")
                if eid and eid not in entities:
                    entities[eid] = {"name": val.strip(), "description": ""}
    return fact_strings, entities


async def execute_cypher_retrieval(
    graph_store: Any,
    llm: Any,
    question: str,
    *,
    max_retries: int = 3,
) -> tuple[list[str], dict[str, dict], dict[str, Any]]:
    """Full text-to-cypher retrieval: generate -> validate -> execute -> parse.

    Results are intended to go DIRECTLY to the final LLM context
    (as a dedicated "Cypher Query Results" section), NOT through
    the cosine reranker.

    Returns:
        fact_strings: Formatted rows from Cypher execution.
        entities: Dict of entity_id -> {name, description}.
        metadata: Dict capturing diagnostic signal — ``cypher`` (final query
            executed), ``cypher_rows`` (row count), ``cypher_fallback``
            (``"label_widened"`` if the 0-row fallback fired, else ``None``).

    On any failure, returns ``([], {}, metadata)`` — never raises.
    """
    metadata: dict[str, Any] = {
        "cypher": None,
        "cypher_rows": 0,
        "cypher_fallback": None,
    }

    cypher = await generate_cypher(llm, question, max_retries=max_retries)
    if not cypher:
        return [], {}, metadata
    metadata["cypher"] = cypher

    try:
        result = await graph_store.query_raw(cypher)
    except Exception as exc:
        logger.debug("Cypher execution failed: %s — query: %s", exc, cypher)
        return [], {}, metadata

    # 0-row recovery: try once with typed labels widened to __Entity__.
    # The LLM's structural reasoning (joins, filters, aggregations) is preserved
    # — only the label predicate is relaxed. No second LLM call.
    if not result.result_set and _should_widen_labels(cypher):
        widened = _widen_typed_labels(cypher)
        if widened != cypher:
            logger.debug("Cypher 0-row retry with widened labels: %s", widened[:120])
            try:
                widened_result = await graph_store.query_raw(widened)
            except Exception as exc:
                logger.debug("Widened cypher execution failed: %s", exc)
            else:
                if widened_result.result_set:
                    result = widened_result
                    metadata["cypher"] = widened
                    metadata["cypher_fallback"] = "label_widened"

    if not result.result_set:
        return [], {}, metadata

    fact_strings, entities = _parse_cypher_result_set(result.result_set)
    metadata["cypher_rows"] = len(fact_strings)

    logger.debug(
        "Cypher retrieval: %d facts, %d entities from: %s",
        len(fact_strings),
        len(entities),
        (metadata["cypher"] or "")[:120],
    )
    return fact_strings, entities, metadata
