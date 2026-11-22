# GraphRAG SDK — Core: an entity's description as pieces in one string
#
# What each source said about an entity is one *piece*. An entity (and a
# MENTIONED_IN edge) stores its pieces in ONE property, ``description``, joined
# with ``" | "``: the text the fulltext index, search, the prompts, the dedup
# judge and the tools read. A piece this version writes never contains ``|``
# — every writer replaces one with ``" ¦ "`` — so a description of several
# pieces splits back into them exactly at ``|``. Such a description also
# carries ``description_count``, the number of pieces: a small integer, not a
# second copy of the text, and what marks the string as one to split.
#
# Text without a count is read whole. A one-piece description needs no count:
# it has no ``|`` to split at. An earlier version's text has none either, and
# there a ``|`` can belong to the text itself (a table row's evidence,
# "CEO | founder of Acme"), so it must not be split:
#
# - with an earlier version's ``descriptions`` list, each list item is one
#   piece, and text the string holds beyond the list (a list left out of step
#   with it) one more;
# - without one, the whole string is one piece.
#
# Each such piece has its ``|`` replaced by the same rule a writer applies, so
# the same text gives the same piece wherever it was read or written: on the
# node, on a chunk's edge, or in a cleanup recorded before the upgrade. The
# next write of a description stores it in this version's form.
#
# The Cypher form of the same rules is in ``storage/merge_query.py``
# (``description_members``, ``joined_description``).

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: Joins two pieces in a stored description.
DESCRIPTION_SEPARATOR = " | "
#: The character that separates pieces in a description that has a count.
PIECE_BREAK = "|"
#: How many pieces a description holds, stored only when it holds two or more.
DESCRIPTION_COUNT_KEY = "description_count"
#: An earlier version's list of the pieces, kept beside ``description``. Read
#: where a graph still has one; never written, and never copied from a merged
#: duplicate onto its survivor (read first, it would hide the survivor's own).
LEGACY_LIST_KEY = "descriptions"
#: What a ``|`` inside a piece becomes (U+00A6, broken bar), with a space each
#: side: it reads the same in a table row's evidence, and the fulltext index
#: splits words at spaces but not at "¦" (so "|Tim Cook|CEO|" written as
#: "¦Tim Cook¦CEO¦" hid "Tim" and "CEO").
PIPE_STAND_IN = "¦"
#: Turned into spaces before a piece is trimmed, here and in Cypher.
_SPACE_LIKE = ("\r", "\n", "\t")


def _tidy(text: str) -> str:
    """Carriage returns, newlines and tabs made spaces, then spaces trimmed
    off both ends. Cypher's ``trim`` is not Python's ``strip()``, so the rule
    is spelled out alike here and in ``merge_query.description_members``."""
    for ch in _SPACE_LIKE:
        text = text.replace(ch, " ")
    return text.strip(" ")


def description_piece(text: Any) -> str:
    """``text`` as one piece: carriage returns, newlines and tabs made
    spaces; ``" | "``, then any other ``|``, replaced with ``" ¦ "``; spaces
    trimmed off both ends. The same rule reads an earlier version's text,
    here and in Cypher, so one text always gives one piece."""
    text = str(text if text is not None else "")
    for ch in _SPACE_LIKE:
        text = text.replace(ch, " ")
    stand_in = f" {PIPE_STAND_IN} "
    return text.replace(DESCRIPTION_SEPARATOR, stand_in).replace(PIECE_BREAK, stand_in).strip(" ")


def _unique(pieces: Iterable[Any], rule: Callable[[str], str] = description_piece) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        cleaned = rule(str(piece)) if piece is not None else ""
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def split_description(text: Any) -> list[str]:
    """The pieces of a description stored with a count, in order, each once,
    blanks left out. ``None`` or an empty string has none."""
    if text is None:
        return []
    return _unique(str(text).split(PIECE_BREAK), _tidy)


def join_description(pieces: Iterable[Any]) -> str:
    """``pieces`` as one stored description: each made a piece
    (:func:`description_piece`), kept once, joined with ``" | "``. A caller
    that stores it also stores the count (:func:`set_description`)."""
    return DESCRIPTION_SEPARATOR.join(_unique(pieces))


def description_count(pieces: list[str]) -> int | None:
    """The ``description_count`` to store with ``pieces``: their number when
    two or more, else none (one piece has nothing to split)."""
    return len(pieces) if len(pieces) > 1 else None


def _earlier_text(description: Any, legacy: Any) -> list[str]:
    """What an earlier version's ``description`` and ``descriptions`` list
    hold, each item whole: the list's items where it has text, then what the
    string holds beyond them; else the whole string."""
    text = None if description is None else str(description)
    if isinstance(legacy, list):
        items = [str(d) for d in legacy if d is not None]
        if any(_tidy(d) for d in items):
            joined = DESCRIPTION_SEPARATOR.join(items)
            if text is None or text == joined:
                return items
            prefix = joined + DESCRIPTION_SEPARATOR
            return [*items, text[len(prefix) :] if text.startswith(prefix) else text]
    return [] if text is None else [text]


def description_pieces(props: Mapping[str, Any]) -> list[str]:
    """The pieces a node's or edge's properties hold: with a
    ``description_count``, ``description`` split at ``|``; without one, the
    text as an earlier version or a single piece left it, each item whole
    (see the module notes)."""
    if props.get(DESCRIPTION_COUNT_KEY) is not None:
        return split_description(props.get("description"))
    return _unique(_earlier_text(props.get("description"), props.get(LEGACY_LIST_KEY)))


def set_description(props: dict[str, Any], pieces: Iterable[Any]) -> list[str]:
    """Store ``pieces`` on ``props``: ``description`` (removed when there are
    none) and its ``description_count`` (when two or more); an earlier
    version's ``descriptions`` list goes. Returns the pieces stored."""
    stored = _unique(pieces)
    props.pop(LEGACY_LIST_KEY, None)
    props.pop(DESCRIPTION_COUNT_KEY, None)
    if stored:
        props["description"] = DESCRIPTION_SEPARATOR.join(stored)
        count = description_count(stored)
        if count is not None:
            props[DESCRIPTION_COUNT_KEY] = count
    else:
        props.pop("description", None)
    return stored
