# GraphRAG SDK — Agentic Retrieval: read-only Cypher guard
# Schema-aware validation for model-generated Cypher, plus the helpers that
# bound what a query may return. The database also runs these queries in
# read-only mode (GRAPH.RO_QUERY); this layer adds the checks the database
# cannot make: labels and properties that exist, typed RELATES edges, no
# procedure calls, no embedding vectors, and a hard row cap.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from graphrag_sdk.core.models import RESERVED_NODE_LABELS, Ontology

logger = logging.getLogger(__name__)

# String literals and backtick identifiers, stripped before keyword scans so
# data values such as 'call center' or `delete_log` cannot trip a guard.
_QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")
_STRING_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_WRITE_RE = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|foreach|call|load\s+csv)\b",
    re.IGNORECASE,
)
_READ_START = ("MATCH", "OPTIONAL MATCH", "UNWIND", "WITH", "RETURN")
_LIMIT_RE = re.compile(r"(?i)\bLIMIT\s+(\d+)")
_EMBEDDING_RE = re.compile(r"\b\w*embedding\w*\b", re.IGNORECASE)
_INTERNAL_LABEL_RE = re.compile(r"__[A-Za-z0-9]+__")
# ``:Name`` plus any ``|Other`` alternatives (``[r:A|B]``). An alternative
# followed by ``.`` is a variable in a list comprehension, not a type.
_LABEL_RE = re.compile(r":\s*`?([A-Za-z_]\w*)`?((?:\s*\|\s*:?\s*`?[A-Za-z_]\w*`?(?!\s*\.))*)")
_ALT_NAME_RE = re.compile(r"\|\s*:?\s*`?([A-Za-z_]\w*)`?")
_SHORTEST_PATH_RE = re.compile(r"\b(allShortestPaths|shortestPath)\s*\(", re.IGNORECASE)

#: Labels that are never a legitimate analytical target (SDK bookkeeping).
INTERNAL_NODE_DENYLIST = ("__GraphRAGConfig__",)

#: Edge types that record SDK bookkeeping rather than a fact between entities.
STRUCTURAL_RELATIONSHIPS = frozenset(
    {"MENTIONED_IN", "MENTIONS", "PART_OF", "NEXT_CHUNK", "SAME_AS", "DISTINCT_FROM"}
)

#: Properties every entity / RELATES edge carries regardless of the ontology.
_SYSTEM_NODE_PROPERTIES = frozenset({"id", "name", "description"})
_SYSTEM_EDGE_PROPERTIES = frozenset({"rel_type", "fact", "src_name", "tgt_name", "weight"})


@dataclass
class GraphSchema:
    """What a generated query may reference.

    ``labels`` / ``relationship_types`` / ``property_keys`` come from the live
    graph (``db.labels()`` and friends). ``semantic_relations`` are the
    ontology's relation labels, stored as ``RELATES {rel_type: ...}``.
    """

    labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    property_keys: list[str] = field(default_factory=list)
    semantic_relations: list[str] = field(default_factory=list)

    def allowed_names(self) -> set[str]:
        """Node labels and relationship types a ``:Name`` token may use."""
        return (
            set(self.labels)
            | set(self.relationship_types)
            | {"__Entity__"}
            | set(RESERVED_NODE_LABELS)
        )


async def introspect_schema(graph_store: Any, ontology: Ontology | None = None) -> GraphSchema:
    """Read the live labels, relationship types and property keys of the graph.

    A procedure that fails leaves its list empty; validation then skips the
    checks that list feeds rather than rejecting every query.
    """

    async def _names(procedure: str) -> list[str]:
        try:
            result = await graph_store.query_raw(f"CALL {procedure}", read_only=True)
        except TypeError:  # graph stores without the read_only keyword
            result = await graph_store.query_raw(f"CALL {procedure}")
        except Exception as exc:
            logger.warning("Schema introspection %s failed: %s", procedure, exc)
            return []
        rows = getattr(result, "result_set", None) or []
        return [str(row[0]) for row in rows if row and row[0]]

    labels = await _names("db.labels()")
    relationship_types = await _names("db.relationshipTypes()")
    property_keys = await _names("db.propertyKeys()")
    semantic = [r.label for r in ontology.relations] if ontology is not None else []
    return GraphSchema(
        labels=labels,
        relationship_types=relationship_types,
        property_keys=property_keys,
        semantic_relations=semantic,
    )


def _strip_maps(text: str) -> str:
    """Remove map literals (``{key: value}``), innermost first, so map keys
    are not mistaken for ``:Label`` tokens."""
    while True:
        text, n = re.subn(r"\{[^{}]*\}", "", text)
        if not n:
            return text


def validate_read_query(cypher: str, schema: GraphSchema | None = None) -> list[str]:
    """Return the reasons ``cypher`` may not run; an empty list means it may.

    Checks, in order: one statement that starts with a read clause, no write
    keyword or procedure call outside string literals, a RETURN clause, no
    embedding properties, no SDK-internal labels, and (with a ``schema``)
    that every label, relationship type and property exists and every
    ``RELATES`` edge constrains its ``rel_type`` to a declared relation.
    """
    errors: list[str] = []
    stripped = cypher.strip().rstrip(";").strip()
    if not stripped:
        return ["the query is empty"]
    no_quotes = _QUOTED_RE.sub(" ", stripped)
    no_strings = _STRING_RE.sub("''", stripped)
    upper = no_quotes.upper().lstrip()
    if not upper.startswith(_READ_START):
        errors.append("the query must start with MATCH, OPTIONAL MATCH, UNWIND, WITH or RETURN")
    if ";" in no_quotes:
        errors.append("only one statement is allowed")
    write = _WRITE_RE.search(no_quotes)
    if write:
        errors.append(f"'{write.group(1).upper()}' is not allowed (read-only, no procedures)")
    if not re.search(r"\bRETURN\b", no_quotes, re.IGNORECASE):
        errors.append("the query must have a RETURN clause")
    if _EMBEDDING_RE.search(_STRING_RE.sub(" ", stripped)):
        errors.append("embedding properties may not be read")
    for internal in INTERNAL_NODE_DENYLIST:
        if internal in no_strings:
            errors.append(f"'{internal}' is internal and may not be queried")
    if schema is None:
        return errors

    scan = _strip_maps(_STRING_RE.sub("''", stripped))
    allowed = schema.allowed_names()
    if schema.labels or schema.relationship_types:
        # With strings and map literals gone, every remaining ``:Name`` is a
        # node label or a relationship type (``(n:A:B)``, ``[r:T|U]``, ``n:A``).
        for first, rest in _LABEL_RE.findall(scan):
            for name in [first, *_ALT_NAME_RE.findall(rest)]:
                # A declared relation used as an edge type gets the more
                # useful "stored as RELATES" hint below instead.
                if name not in allowed and name not in schema.semantic_relations:
                    errors.append(f"unknown label or relationship type '{name}'")
    if schema.property_keys:
        known = set(schema.property_keys) | _SYSTEM_NODE_PROPERTIES | _SYSTEM_EDGE_PROPERTIES
        for prop in sorted(set(re.findall(r"\b[A-Za-z_]\w*\s*\.\s*`?([A-Za-z_]\w*)`?", scan))):
            if prop not in known:
                errors.append(f"unknown property '{prop}'")
    if schema.semantic_relations:
        declared = set(schema.semantic_relations)
        for variable in re.findall(r"\[\s*([A-Za-z_]\w*)?\s*:\s*`?RELATES`?", scan):
            if not variable:
                errors.append("every RELATES edge must bind a variable and constrain its rel_type")
                continue
            values = re.findall(
                rf"\b{re.escape(variable)}\s*\.\s*rel_type\s*(?:=|IN)\s*(\[[^\]]*\]|'[^']*'|\"[^\"]*\")",
                stripped,
                re.IGNORECASE,
            )
            if not values:
                errors.append(
                    f"RELATES edge '{variable}' must constrain {variable}.rel_type to one of: "
                    f"{', '.join(sorted(declared))}"
                )
            for value in values:
                for rel in re.findall(r"['\"]([^'\"]+)['\"]", value):
                    if rel not in declared:
                        errors.append(f"unknown relation '{rel}' in rel_type")
        for rel in re.findall(r"\[\s*(?:[A-Za-z_]\w*)?\s*:\s*`?([A-Za-z_]\w*)`?", scan):
            if rel in declared and rel not in schema.relationship_types:
                errors.append(
                    f"'{rel}' is stored as RELATES; match [r:RELATES] with r.rel_type = '{rel}'"
                )
    # Keep the order stable but drop exact repeats.
    return list(dict.fromkeys(errors))


def enforce_row_cap(cypher: str, cap: int) -> str:
    """Bound every ``LIMIT`` to ``cap`` and add one when the query has none.

    Every ``LIMIT n`` is rewritten to ``min(n, cap)``, not just the last, so
    an inner ``WITH ... LIMIT 100000`` cannot build a huge intermediate set.
    ``shortestPath`` wrappers are removed (FalkorDB does not support them in
    this form).
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")
    cypher = _SHORTEST_PATH_RE.sub("(", cypher.strip().rstrip(";"))
    capped, n = _LIMIT_RE.subn(lambda m: f"LIMIT {min(int(m.group(1)), cap)}", cypher)
    if n == 0:
        return f"{capped}\nLIMIT {cap}"
    return capped


def mask_internal_labels(text: str) -> str:
    """Replace SDK-internal ``__Label__`` names in output with ``[internal]``."""
    return _INTERNAL_LABEL_RE.sub("[internal]", text)
