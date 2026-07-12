# GraphRAG SDK — Tools: read-only Cypher guard
# Fail-closed validation for agent-supplied Cypher. Strings/comments are
# removed with a single-pass lexer state machine (ordering regex passes is
# exploitable), then write keywords are scanned on both the raw and the
# NFKC-normalized copies. The ORIGINAL query text is what gets executed.

from __future__ import annotations

import re
import unicodedata

from graphrag_sdk.core.exceptions import ReadOnlyViolation

_WRITE_TOKENS = ("CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "FOREACH")
_START_KEYWORDS = ("MATCH", "OPTIONAL", "UNWIND", "WITH", "RETURN", "CALL")
# Full procedure names only — prefixes are unsafe (db.idx.fulltext.createNodeIndex
# is a WRITE). Compared lowercase.
READ_SAFE_PROCEDURES = frozenset(
    {
        "db.labels",
        "db.relationshiptypes",
        "db.propertykeys",
        "db.indexes",
        "db.idx.fulltext.querynodes",
        "db.idx.fulltext.queryrelationships",
        "db.idx.vector.querynodes",
        "db.idx.vector.queryrelationships",
    }
)


def _strip_noise(text: str) -> str:
    """Remove comments and mask string/backtick literals in one lexer pass.

    Comments are removed as EMPTY string (not a space) on purpose: it is
    fail-closed against mid-token splitting (``Cr/**/eate`` reassembles to
    ``Create`` and is caught), at the cost of over-rejecting queries that
    rely on a comment as the only token separator — acceptable for a guard.
    String/backtick literals become single spaces so their content can never
    trip (or hide) a keyword.
    """
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block"
                i += 2
                continue
            if ch in ("'", '"', "`"):
                state = ch
                out.append(" ")
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if state == "line":
            if ch == "\n":
                state = "code"
                out.append("\n")
            i += 1
            continue
        if state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
            i += 1
            continue
        # inside a quoted literal (state is the quote char)
        if ch == "\\" and state in ("'", '"'):
            i += 2
            continue
        if ch == state:
            state = "code"
            out.append(" ")
        i += 1
    return "".join(out)


def _scan(stripped: str) -> None:
    """Reject write constructs in noise-stripped Cypher text."""
    first = re.match(r"\s*([A-Za-z]+)", stripped)
    if first and first.group(1).upper() not in _START_KEYWORDS:
        raise ReadOnlyViolation(
            f"Query must start with one of {', '.join(_START_KEYWORDS)}; got '{first.group(1)}'.",
            offending_token=first.group(1),
        )
    body = stripped.rstrip().rstrip(";")
    if ";" in body:
        raise ReadOnlyViolation("Multiple Cypher statements are not allowed.", offending_token=";")
    for token in _WRITE_TOKENS:
        if re.search(rf"\b{token}\b", stripped, re.IGNORECASE):
            raise ReadOnlyViolation(
                f"Write operation '{token}' is not allowed — cypher_read is read-only.",
                offending_token=token,
            )
    if re.search(r"\bLOAD\s+CSV\b", stripped, re.IGNORECASE):
        raise ReadOnlyViolation("LOAD CSV is not allowed.", offending_token="LOAD CSV")
    for match in re.finditer(r"\bCALL\b(\s*)([A-Za-z0-9_.]*)", stripped, re.IGNORECASE):
        proc = match.group(2)
        if not proc:
            rest = stripped[match.end() :].lstrip()
            if rest.startswith("{"):
                continue  # CALL { subquery }: inner writes caught by the scan above
            raise ReadOnlyViolation("Bare CALL is not allowed.", offending_token="CALL")
        if proc.lower() not in READ_SAFE_PROCEDURES:
            raise ReadOnlyViolation(
                f"Procedure '{proc}' is not on the read-safe allowlist "
                f"({', '.join(sorted(READ_SAFE_PROCEDURES))}).",
                offending_token=f"CALL {proc}",
            )


def ensure_read_only(query: str) -> None:
    """Raise :class:`ReadOnlyViolation` unless *query* is a read-only statement."""
    if not query or not query.strip():
        raise ReadOnlyViolation("Empty Cypher query.", offending_token=None)
    _scan(_strip_noise(query))
    _scan(_strip_noise(unicodedata.normalize("NFKC", query)))


def apply_limit(query: str, limit: int) -> tuple[str, bool]:
    """Append ``LIMIT {limit}`` when the query has no LIMIT clause.

    LIMIT detection runs on the noise-stripped copy so a literal string
    containing the word LIMIT does not suppress injection. Returns the
    (possibly rewritten) query and whether injection happened.
    """
    trimmed = query.rstrip().rstrip(";")
    if re.search(r"\bLIMIT\b", _strip_noise(trimmed), re.IGNORECASE):
        return trimmed, False
    return f"{trimmed}\nLIMIT {int(limit)}", True
