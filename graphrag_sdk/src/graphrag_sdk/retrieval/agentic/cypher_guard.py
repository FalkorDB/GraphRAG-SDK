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
_LABEL_RE = re.compile(r":\s*`?([A-Za-z_]\w*)`?((?:\s*\|\s*:?\s*`?[A-Za-z_]\w*\b`?(?!\s*\.))*)")
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


@dataclass
class _Scanned:
    """A query split into what runs and what is data.

    ``code`` is the query with comments removed (strings intact): what is
    actually executed. ``blank`` has the same length as ``code`` with the
    contents of string literals replaced by spaces and backtick identifiers
    reduced to word characters, so keyword / label / property scans see
    only real syntax, and offsets in ``blank`` are offsets in ``code``.
    """

    code: str
    blank: str
    error: str = ""


def _scan(cypher: str) -> _Scanned:
    code: list[str] = []
    blank: list[str] = []
    i, n = 0, len(cypher)
    while i < n:
        ch = cypher[i]
        nxt = cypher[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            end = cypher.find("\n", i)
            i = n if end == -1 else end
            code.append(" ")
            blank.append(" ")
            continue
        if ch == "/" and nxt == "*":
            end = cypher.find("*/", i + 2)
            if end == -1:
                return _Scanned(
                    "".join(code), "".join(blank), "the query has an unterminated comment"
                )
            i = end + 2
            code.append(" ")
            blank.append(" ")
            continue
        if ch in ("'", '"'):
            j = i + 1
            while j < n and cypher[j] != ch:
                j += 2 if cypher[j] == "\\" else 1
            if j >= n:
                return _Scanned(
                    "".join(code), "".join(blank), "the query has an unterminated string"
                )
            literal = cypher[i : j + 1]
            code.append(literal)
            blank.append(ch + " " * (len(literal) - 2) + ch)
            i = j + 1
            continue
        if ch == "`":
            j = cypher.find("`", i + 1)
            if j == -1:
                return _Scanned(
                    "".join(code), "".join(blank), "the query has an unterminated identifier"
                )
            literal = cypher[i : j + 1]
            code.append(literal)
            blank.append("`" + re.sub(r"\W", "_", literal[1:-1]) + "`")
            i = j + 1
            continue
        code.append(ch)
        blank.append(ch)
        i += 1
    return _Scanned("".join(code), "".join(blank))


def _strip_maps(text: str) -> str:
    """Remove map literals (``{key: value}``), innermost first, so map keys
    are not mistaken for ``:Label`` tokens."""
    while True:
        text, n = re.subn(r"\{[^{}]*\}", "", text)
        if not n:
            return text


#: Words after which ``[`` opens a list literal or comprehension, not a subscript.
_LIST_CONTEXT_WORDS = frozenset(
    {
        "IN", "RETURN", "WITH", "UNWIND", "WHERE", "AND", "OR", "XOR", "NOT", "AS",
        "CASE", "WHEN", "THEN", "ELSE", "BY", "DISTINCT", "CONTAINS", "IS", "NULL",
        "STARTS", "ENDS", "LIMIT", "SKIP", "ORDER", "YIELD",
    }
)  # fmt: skip
_SUBSCRIPT_RE = re.compile(r"(`[^`]*`|[A-Za-z_]\w*|[)\]])\s*\[")
_LITERAL_INDEX_RE = re.compile(r"\s*-?\d*\s*(?:\]|\.\.)")


def _dynamic_subscripts(blank: str) -> list[str]:
    """Subscripts whose index is not an integer literal (``n['embedding']``,
    ``n[k]``, ``properties(n)[$p]``): they read properties by computed name,
    which no static check can follow."""
    found = []
    for match in _SUBSCRIPT_RE.finditer(blank):
        target = match.group(1)
        # A keyword opens a list; a backticked name is always an identifier.
        if not target.startswith("`") and target.upper() in _LIST_CONTEXT_WORDS:
            continue
        if not _LITERAL_INDEX_RE.match(blank, match.end()):
            found.append(target)
    return found


def validate_read_query(cypher: str, schema: GraphSchema | None = None) -> list[str]:
    """Return the reasons ``cypher`` may not run; an empty list means it may.

    The query is tokenized first (strings, backtick identifiers, ``//`` and
    ``/* */`` comments), so nothing can hide behind a comment or a quote.
    Checks: one statement that starts with a read clause; no write keyword
    or procedure call; a RETURN clause; no embedding properties and no
    property read by computed name (``n['x']``, ``n[k]``); no SDK-internal
    labels anywhere in the text; and (with a ``schema``) that every label,
    relationship type and property exists and every ``RELATES`` edge
    constrains its ``rel_type`` to a declared relation.
    """
    errors: list[str] = []
    scanned = _scan(cypher.strip())
    if scanned.error:
        return [scanned.error]
    code = scanned.code.strip().rstrip(";").strip()
    blank = scanned.blank.strip().rstrip(";").strip()
    if not blank:
        return ["the query is empty"]
    if not blank.upper().startswith(_READ_START):
        errors.append("the query must start with MATCH, OPTIONAL MATCH, UNWIND, WITH or RETURN")
    if ";" in blank:
        errors.append("only one statement is allowed")
    write = _WRITE_RE.search(blank)
    if write:
        errors.append(f"'{write.group(1).upper()}' is not allowed (read-only, no procedures)")
    if not re.search(r"\bRETURN\b", blank, re.IGNORECASE):
        errors.append("the query must have a RETURN clause")
    if _EMBEDDING_RE.search(blank):
        errors.append("embedding properties may not be read")
    dynamic = _dynamic_subscripts(_strip_maps(blank))
    if dynamic:
        errors.append(
            "properties may not be read by computed name (e.g. n['x'] or n[k]); use n.property"
        )
    for internal in INTERNAL_NODE_DENYLIST:
        # Checked on the raw text: a label named inside a string
        # (``'__GraphRAGConfig__' IN labels(n)``) selects the node just the same.
        if internal in cypher:
            errors.append(f"'{internal}' is internal and may not be queried")
    if schema is None:
        return list(dict.fromkeys(errors))

    scan = _strip_maps(blank)
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
                code,
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
    """Return the query without comments, every ``LIMIT`` bounded to ``cap``.

    Every real ``LIMIT n`` clause is rewritten to ``min(n, cap)``, not just
    the last, so an inner ``WITH ... LIMIT 100000`` cannot build a huge
    intermediate set; ``LIMIT`` inside a string or a comment is not a
    clause and is left alone (comments are dropped). A query without a
    ``LIMIT`` gets one. ``shortestPath`` wrappers are removed (FalkorDB does
    not support them in this form).
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")
    scanned = _scan(cypher.strip())
    if scanned.error:
        raise ValueError(scanned.error)
    code, blank = scanned.code, scanned.blank
    pieces: list[str] = []
    last = 0
    found = 0
    for match in _LIMIT_RE.finditer(blank):
        found += 1
        pieces.append(code[last : match.start()])
        pieces.append(f"LIMIT {min(int(match.group(1)), cap)}")
        last = match.end()
    pieces.append(code[last:])
    capped = "".join(pieces).strip().rstrip(";").rstrip()
    capped = _SHORTEST_PATH_RE.sub("(", capped)
    if found == 0:
        return f"{capped}\nLIMIT {cap}"
    return capped


def mask_internal_labels(text: str) -> str:
    """Replace SDK-internal ``__Label__`` names in output with ``[internal]``."""
    return _INTERNAL_LABEL_RE.sub("[internal]", text)
