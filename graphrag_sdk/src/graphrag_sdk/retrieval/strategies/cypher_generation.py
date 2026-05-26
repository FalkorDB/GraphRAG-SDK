# GraphRAG SDK — Retrieval: Text-to-Cypher Generation
# Adapted from FalkorDB/GraphRAG-SDK upstream for the unified RELATES ontology.
# Generates read-only Cypher queries from natural language questions.
#
# Design: Uses typed entity node labels (Person, Organization, Location, etc.)
# for routing and aggregation rather than matching rel_type on RELATES edges.
# Cypher results go directly to the LLM context — NOT through the reranker.

from __future__ import annotations

import logging
import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import Ontology

logger = logging.getLogger(__name__)

# ── Valid labels for our graph ontology ────────────────────────────

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

_RESERVED_ENTITY_PROPS = (
    ("name", "STRING", "entity name"),
    ("description", "STRING", "entity description"),
)
_RESERVED_REL_PROPS = (
    ("rel_type", "STRING", "original relation type as a string"),
    ("fact", "STRING", "evidence text for the relation"),
    ("src_name", "STRING", "source entity name"),
    ("tgt_name", "STRING", "target entity name"),
)

_NUMERIC_TYPES = frozenset({"INTEGER", "FLOAT"})

_WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|CALL\s+db\.idx)\b",
    re.IGNORECASE,
)


def _labels_from_ontology(ontology: Ontology | None) -> frozenset[str]:
    """Return the entity labels declared in ``ontology``, falling back to the
    historical hardcoded set when the ontology is empty (open-ontology mode).
    """
    if ontology is None or not ontology.entities:
        return _ENTITY_LABELS
    return frozenset(e.label for e in ontology.entities)


def render_ontology_block(ontology: Ontology | None) -> str:
    """Render a Markdown ontology block listing declared entity / relation
    properties, derived from the live ``Ontology``.

    Mirrors LangChain's ``Neo4jGraph.get_schema()`` and LlamaIndex's
    ``Neo4jPropertyGraphStore.get_schema_str()``. Always emits the reserved
    SDK keys (``name``, ``description`` on entities; ``rel_type``, ``fact``,
    ``src_name``, ``tgt_name`` on RELATES) so the LLM still knows about them
    even when the ontology declares no custom properties.
    """
    labels: list[str]
    rel_labels: list[str]
    ent_props: dict[str, list[tuple[str, str, str | None]]] = {}
    rel_props: dict[str, list[tuple[str, str, str | None]]] = {}

    if ontology is not None and ontology.entities:
        labels = [e.label for e in ontology.entities]
        for e in ontology.entities:
            ent_props[e.label] = [(p.name, p.type, p.description) for p in e.properties]
    else:
        labels = sorted(_ENTITY_LABELS)

    if ontology is not None and ontology.relations:
        rel_labels = [r.label for r in ontology.relations]
        for r in ontology.relations:
            rel_props[r.label] = [(p.name, p.type, p.description) for p in r.properties]
    else:
        rel_labels = []

    lines: list[str] = []
    lines.append("### Entity node labels (all entities also carry `__Entity__`):")
    for label in labels:
        lines.append(f"- {label}")
        lines.append("    Properties:")
        for name, typ, desc in _RESERVED_ENTITY_PROPS:
            lines.append(f"      - {name} ({typ}) — {desc}")
        for name, typ, desc in ent_props.get(label, []):
            d = f" — {desc}" if desc else ""
            lines.append(f"      - {name} ({typ}){d}")

    lines.append("")
    lines.append("### Edge types:")
    lines.append("- RELATES: connects any entity to any entity.")
    lines.append("    Properties:")
    for name, typ, desc in _RESERVED_REL_PROPS:
        lines.append(f"      - {name} ({typ}) — {desc}")
    union_rel_props: dict[str, tuple[str, str | None]] = {}
    for label in rel_labels:
        for name, typ, desc in rel_props.get(label, []):
            if name not in union_rel_props:
                union_rel_props[name] = (typ, desc)
    for name, (typ, desc) in union_rel_props.items():
        d = f" — {desc}" if desc else ""
        lines.append(f"      - {name} ({typ}){d}  # declared on RELATES via rel_type filters")
    if rel_labels:
        lines.append("    Allowed `rel_type` values: " + ", ".join(rel_labels))
    lines.append("- MENTIONED_IN: connects entity to Chunk node (provenance)")
    lines.append("- PART_OF: connects Document to Chunk")
    lines.append("- NEXT_CHUNK: connects Chunk to next sequential Chunk")
    return "\n".join(lines)


def _render_attribute_examples(ontology: Ontology | None) -> str:
    """Synthesize one filter example per declared numeric attribute.

    Helps the LLM learn that custom numeric properties exist and can be used
    in ``WHERE`` / ``ORDER BY`` / aggregations. Returns ``""`` when no
    numeric attributes are declared.
    """
    if ontology is None:
        return ""
    examples: list[str] = []
    for et in ontology.entities:
        for p in et.properties:
            if p.type in _NUMERIC_TYPES and len(examples) < 2:
                var = et.label[0].lower()
                examples.append(
                    f'Question: "Which {et.label} has the highest {p.name}?"\n'
                    f"```cypher\n"
                    f"MATCH ({var}:{et.label})\n"
                    f"WHERE {var}.{p.name} IS NOT NULL\n"
                    f"RETURN {var}.name AS name, {var}.{p.name} AS {p.name}\n"
                    f"ORDER BY {var}.{p.name} DESC\n"
                    f"LIMIT 10\n"
                    f"```"
                )
    if not examples:
        return ""
    # No leading newline — the template places ``{attribute_examples}`` on its
    # own line already, so this block stays cleanly separated from the closing
    # code fence above it.
    return "\n\n".join(examples)


# ── Schema prompt ────────────────────────────────────────────────

_ONTOLOGY_PROMPT_TEMPLATE = """\
You are a Cypher query generator for a FalkorDB graph database.

## Graph Schema

{ontology_block}

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

Question: "What organizations are related to the technology?"
```cypher
MATCH (o:Organization)-[r:RELATES]-(t:Technology)
RETURN o.name AS organization, t.name AS technology, r.rel_type AS relation, r.fact AS evidence
LIMIT 20
```

{attribute_examples}

## Your task

Generate a single Cypher query to answer the following question.
If you cannot generate a valid query, return an empty code block.
Return ONLY the Cypher query inside triple backticks.

Question: {question}
"""


def build_ontology_prompt(ontology: Ontology | None, question: str) -> str:
    """Build the full Cypher generation prompt for ``question`` from ``ontology``.

    When ``ontology`` is empty, the prompt falls back to the historical
    hardcoded label set and matches today's behavior bit-for-bit aside from
    the new ontology block formatting.
    """
    return _ONTOLOGY_PROMPT_TEMPLATE.format(
        ontology_block=render_ontology_block(ontology),
        attribute_examples=_render_attribute_examples(ontology),
        question=question,
    )


# Backwards-compatible alias for callers that import ONTOLOGY_PROMPT.
# It exposes the template form (still expects ``{ontology_block}``,
# ``{attribute_examples}``, and ``{question}`` placeholders) — direct
# ``.format(question=...)`` callers should migrate to ``build_ontology_prompt``.
ONTOLOGY_PROMPT = _ONTOLOGY_PROMPT_TEMPLATE


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


def _pop_schema_alias(kwargs: dict, ontology: Ontology | None, func: str) -> Ontology | None:
    """Back-compat helper: pop a legacy ``schema=`` kwarg, warn, and return
    the resolved ``ontology`` value. Raises if both names were supplied.
    """
    if "schema" in kwargs:
        import warnings

        legacy = kwargs.pop("schema")
        if kwargs:
            raise TypeError(f"{func}() got unexpected keyword arguments: {sorted(kwargs)}")
        if ontology is not None:
            raise TypeError(
                f"{func}() received both `ontology=` and `schema=`. "
                f"Use `ontology=` only; `schema=` is deprecated."
            )
        warnings.warn(
            f"The `schema=` keyword argument on {func}() has been renamed "
            f"to `ontology=` (graphrag_sdk v1.2+). Update your call site "
            f"— the alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy
    if kwargs:
        raise TypeError(f"{func}() got unexpected keyword arguments: {sorted(kwargs)}")
    return ontology


def validate_cypher(cypher: str, ontology: Ontology | None = None, **legacy: Any) -> list[str]:
    """Validate generated Cypher for safety and correctness.

    Uses an allowlist approach: the query must start with a read-only
    keyword, and dangerous constructs are explicitly rejected.

    When ``ontology`` is provided, label validation uses the labels declared
    in the ontology (plus structural labels); otherwise it falls back to the
    historical hardcoded label set.

    Returns list of error strings; empty list means valid.
    """
    ontology = _pop_schema_alias(legacy, ontology, "validate_cypher")
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

    # Allowlist: must start with a read-only keyword
    if not re.match(r"^(MATCH|OPTIONAL\s+MATCH|UNWIND|WITH)\b", cypher_norm, re.IGNORECASE):
        errors.append("Query must start with MATCH, OPTIONAL MATCH, UNWIND, or WITH")

    # Reject multi-statement queries
    if ";" in cypher_norm:
        errors.append("Multiple Cypher statements are not allowed")

    # Reject procedures and bulk import
    if re.search(r"\bCALL\b", cypher_norm, re.IGNORECASE):
        errors.append("CALL procedures are not allowed in generated queries")
    if re.search(r"\bLOAD\s+CSV\b", cypher_norm, re.IGNORECASE):
        errors.append("LOAD CSV is not allowed in generated queries")

    # No write operations
    if _WRITE_KEYWORDS.search(cypher_norm):
        errors.append("Write operation detected — query must be read-only")

    # Must have RETURN
    if not re.search(r"\bRETURN\b", cypher_norm, re.IGNORECASE):
        errors.append("Missing RETURN clause")

    # Check referenced labels exist in ontology
    allowed_labels = _labels_from_ontology(ontology) | _STRUCTURAL_LABELS
    label_pattern = re.findall(r"\((?:\w+)?:(\w+)", cypher_norm)
    for label in label_pattern:
        if label not in allowed_labels:
            errors.append(f"Unknown label: {label}")

    return errors


# ── Text-to-Cypher execution ────────────────────────────────────


async def generate_cypher(
    llm: Any,
    question: str,
    *,
    ontology: Ontology | None = None,
    max_retries: int = 3,
    ctx: Context | None = None,
    **legacy: Any,
) -> str | None:
    """Generate a Cypher query from a natural language question.

    When ``ontology`` is provided, the prompt and validator both use the
    declared labels and properties.

    Returns the Cypher string, or None if all retries fail.
    """
    ontology = _pop_schema_alias(legacy, ontology, "generate_cypher")
    prompt = build_ontology_prompt(ontology, question)
    last_error = ""

    for attempt in range(max_retries):
        try:
            if ctx is not None:
                ctx.ensure_budget("Cypher generation LLM call")
            if attempt > 0 and last_error:
                prompt_with_feedback = (
                    prompt + f"\n\nPrevious attempt failed with error: {last_error}\n"
                    "Remember: no shortestPath, every RETURN column must have a "
                    "unique alias, add LIMIT, keep it simple."
                )
                response = await llm.ainvoke(
                    prompt_with_feedback,
                    timeout=(
                        ctx.provider_timeout_seconds("Cypher generation LLM call")
                        if ctx is not None
                        else None
                    ),
                )
            else:
                response = await llm.ainvoke(
                    prompt,
                    timeout=(
                        ctx.provider_timeout_seconds("Cypher generation LLM call")
                        if ctx is not None
                        else None
                    ),
                )

            cypher = extract_cypher(response.content)
            if not cypher:
                last_error = "Empty query generated"
                continue

            errors = validate_cypher(cypher, ontology)
            if errors:
                last_error = "; ".join(errors)
                continue

            # Sanitize before returning
            cypher = _sanitize_cypher(cypher)
            return cypher

        except LatencyBudgetExceededError:
            raise
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
    ontology: Ontology | None = None,
    max_retries: int = 3,
    ctx: Context | None = None,
    **legacy: Any,
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
    ontology = _pop_schema_alias(legacy, ontology, "execute_cypher_retrieval")
    cypher = await generate_cypher(
        llm, question, ontology=ontology, max_retries=max_retries, ctx=ctx
    )
    if not cypher:
        return [], {}

    try:
        if ctx is not None:
            ctx.ensure_budget("Cypher execution")
        result = await graph_store.query_raw(cypher)
    except LatencyBudgetExceededError:
        raise
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
        len(fact_strings),
        len(entities),
        cypher[:120],
    )
    return fact_strings, entities


# ── Deprecation aliases ──────────────────────────────────────────


def _deprecated_build_schema_prompt(ontology: Ontology | None, question: str) -> str:
    """DEPRECATED: use ``build_ontology_prompt`` instead."""
    import warnings

    warnings.warn(
        "`build_schema_prompt` has been renamed to `build_ontology_prompt` "
        "(graphrag_sdk v1.2+). Update your import — the alias will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_ontology_prompt(ontology, question)


def _deprecated_render_schema_block(ontology: Ontology | None) -> str:
    """DEPRECATED: use ``render_ontology_block`` instead."""
    import warnings

    warnings.warn(
        "`render_schema_block` has been renamed to `render_ontology_block` "
        "(graphrag_sdk v1.2+). Update your import — the alias will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return render_ontology_block(ontology)


def __getattr__(name: str):  # PEP 562
    if name == "SCHEMA_PROMPT":
        import warnings

        warnings.warn(
            "`SCHEMA_PROMPT` has been renamed to `ONTOLOGY_PROMPT` "
            "(graphrag_sdk v1.2+). Update your import — the alias will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ONTOLOGY_PROMPT
    if name == "build_schema_prompt":
        return _deprecated_build_schema_prompt
    if name == "render_schema_block":
        return _deprecated_render_schema_block
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
