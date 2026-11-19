# GraphRAG SDK — Core: an entity's description as pieces in one string
#
# What each source said about an entity is one *piece*. An entity (and a
# MENTIONED_IN edge) stores its pieces in ONE property, ``description``, joined
# with ``" | "``: the text the fulltext index, search, the prompts, the dedup
# judge and the tools read. A piece never contains ``|`` — every writer
# replaces one with ``¦`` — so ``|`` in a stored description always separates
# two pieces, and splitting at it gives them back exactly. Nothing stores the
# pieces a second time.
#
# Graphs written by earlier versions kept a ``descriptions`` list beside the
# string. Where a node or edge still has one with text in it, it is read as the
# pieces, and the next write of that description drops it. Reading what an
# earlier version wrote, every ``|`` separates pieces: in a list's item as in a
# description without a list, so a piece that held one (a table row's
# evidence) reads as several, with every word kept. That one rule makes a
# node's list, a one-piece edge's string and a cleanup recorded before the
# upgrade agree, so a delete still finds the text it takes off.
#
# The Cypher form of the same rules is in ``storage/merge_query.py``
# (``description_members``, ``joined_description``).

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

#: Joins two pieces in a stored description.
DESCRIPTION_SEPARATOR = " | "
#: The character that only ever separates pieces.
PIECE_BREAK = "|"
#: An earlier version's list of the pieces, kept beside ``description``. Read
#: where a graph still has one; never written, and never copied from a merged
#: duplicate onto its survivor (read first, it would hide the survivor's own).
LEGACY_LIST_KEY = "descriptions"
#: What a ``|`` inside a piece becomes (U+00A6, broken bar): it reads the same
#: in a table row's evidence and never splits the piece. Written with a space
#: each side: the fulltext index splits words at "|" and at spaces but not at
#: "¦", so "|Tim Cook|CEO|" written as "¦Tim Cook¦CEO¦" hid "Tim" and "CEO".
PIPE_STAND_IN = "\u00a6"
_PIPE_RUN = re.compile(r"\s*\|\s*")


def description_piece(text: Any) -> str:
    """``text`` as one piece: trimmed, each ``|`` (with the spaces around it)
    replaced with ``" ¦ "``."""
    raw = str(text if text is not None else "")
    return _PIPE_RUN.sub(f" {PIPE_STAND_IN} ", raw).strip()


def _unique(pieces: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        cleaned = description_piece(piece)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def split_description(text: Any) -> list[str]:
    """The pieces of a stored description, in order, each once, blanks left
    out. ``None`` or an empty string has none."""
    if text is None:
        return []
    return _unique(str(text).split(PIECE_BREAK))


def join_description(pieces: Iterable[Any]) -> str:
    """``pieces`` as one stored description: each made a piece
    (:func:`description_piece`), kept once, joined with ``" | "``."""
    return DESCRIPTION_SEPARATOR.join(_unique(pieces))


def description_pieces(props: Mapping[str, Any]) -> list[str]:
    """The pieces a node's or edge's properties hold: ``description`` split at
    ``|``, or, where an earlier version's ``descriptions`` list with text is
    present, each of its items split at ``|``."""
    legacy = props.get(LEGACY_LIST_KEY)
    if isinstance(legacy, list):
        items = [str(d) for d in legacy if d is not None]
        if any(d.strip() for d in items):
            return _unique(p for d in items for p in d.split(PIECE_BREAK))
    return split_description(props.get("description"))


def set_description(props: dict[str, Any], pieces: Iterable[Any]) -> list[str]:
    """Store ``pieces`` on ``props`` as its ``description`` — removed when
    there are none — and drop an earlier version's ``descriptions`` list.
    Returns the pieces stored."""
    stored = _unique(pieces)
    props.pop(LEGACY_LIST_KEY, None)
    if stored:
        props["description"] = DESCRIPTION_SEPARATOR.join(stored)
    else:
        props.pop("description", None)
    return stored
