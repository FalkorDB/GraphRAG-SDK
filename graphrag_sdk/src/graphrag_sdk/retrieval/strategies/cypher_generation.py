# GraphRAG SDK 2.0 — Retrieval: Text-to-Cypher Generation
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

_ENTITY_LABELS = frozenset({
    "Person", "Organization", "Technology", "Product", "Location",
    "Date", "Event", "Concept", "Law", "Dataset", "Method",
})

_STRUCTURAL_LABELS = frozenset({"Chunk", "Document", "__Entity__"})

_ALL_LABELS = _ENTITY_LABELS | _STRUCTURAL_LABELS

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|CALL\s+db\.idx)\b",
    re.IGNORECASE,
)

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
1. Do NOT use shortestPath() or allShortestPaths() — FalkorDB returns Path objects that cause "Type mismatch: expected List or Null but was Path".
2. Every column in RETURN must have a UNIQUE name. Use aliases: `RETURN a.name AS a_name, b.name AS b_name` — NEVER `RETURN a.name, b.name` without aliases when both are `.name`.
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
RETURN a.name AS from_entity, r1.rel_type AS rel1, mid.name AS via_entity, r2.rel_type AS rel2, b.name AS to_entity
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

Question: "What organizations are related to the technology?"
```cypher
MATCH (o:Organization)-[r:RELATES]-(t:Technology)
RETURN o.name AS organization, t.name AS technology, r.rel_type AS relation, r.fact AS evidence
LIMIT 20
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

    # Add LIMIT if missing (prevent runaway scans)
    if not re.search(r"\bLIMIT\b", cypher, re.IGNORECASE):
        cypher = cypher.rstrip().rstrip(";") + "\nLIMIT 25"

    return cypher


# ── Cypher validation ────────────────────────────────────────────

def validate_cypher(cypher: str) -> list[str]:
    """Validate generated Cypher for safety and correctness.

    Returns list of error strings; empty list means valid.
    """
    errors: list[str] = []
    if not cypher:
        errors.append("Empty Cypher query")
        return errors

    # No write operations
    if _WRITE_KEYWORDS.search(cypher):
        errors.append("Write operation detected — query must be read-only")

    # Must have RETURN
    if not re.search(r"\bRETURN\b", cypher, re.IGNORECASE):
        errors.append("Missing RETURN clause")

    # Check referenced labels exist in schema
    label_pattern = re.findall(r"\((?:\w+)?:(\w+)", cypher)
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
                    prompt
                    + f"\n\nPrevious attempt failed with error: {last_error}\n"
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


async def execute_cypher_retrieval(
    graph_store: Any,
    llm: Any,
    question: str,
    *,
    max_retries: int = 3,
) -> tuple[list[str], dict[str, dict]]:
    """Full text-to-cypher retrieval: generate -> validate -> execute -> parse.

    Results are intended to go DIRECTLY to the final LLM context
    (as a dedicated "Cypher Query Results" section), NOT through
    the cosine reranker.

    Returns:
        fact_strings: Formatted rows from Cypher execution.
        entities: Dict of entity_id -> {name, description}.

    On any failure, returns empty results (silent degradation).
    """
    cypher = await generate_cypher(llm, question, max_retries=max_retries)
    if not cypher:
        return [], {}

    try:
        result = await graph_store.query_raw(cypher)
    except Exception as exc:
        logger.debug("Cypher execution failed: %s — query: %s", exc, cypher)
        return [], {}

    if not result.result_set:
        return [], {}

    # Parse results into readable fact lines and entity dict
    fact_strings: list[str] = []
    entities: dict[str, dict] = {}

    for row in result.result_set:
        parts: list[str] = []
        for val in row:
            if val is None:
                continue
            s = str(val).strip()
            if s and s != "[]" and s != "{}":
                parts.append(s)
        if not parts:
            continue

        line = " | ".join(parts)
        fact_strings.append(line)

        # Extract entity names (strings that look like names, not numbers/lists)
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

    logger.debug(
        "Cypher retrieval: %d facts, %d entities from: %s",
        len(fact_strings), len(entities), cypher[:120],
    )
    return fact_strings, entities
