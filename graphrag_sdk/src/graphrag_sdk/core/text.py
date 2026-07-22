"""Name-normalization helpers shared by storage and resolution.

Kept in ``core`` so both the graph store (which writes ``name_key`` at ingest)
and the resolution strategies (which look entities up by it) can agree on one
definition without cross-layer imports.
"""

import re

_SEPARATORS = re.compile(r"[\s\-_]+")


def normalize_name(name: str) -> str:
    """Fold case and separator runs so surface variants share one form.

    ``"GraphRAG-SDK"``, ``"graphrag_sdk"`` and ``"GraphRAG SDK"`` all normalize
    to ``"graphrag sdk"``.
    """
    return _SEPARATORS.sub(" ", str(name).strip().lower()).strip()


def name_key(name: str) -> str:
    """A separator-free key for indexed entity lookup.

    Drops the spaces from :func:`normalize_name`, so it is symmetric across
    tokenization variants — ``"llama_index"`` and ``"LlamaIndex"`` both key to
    ``"llamaindex"``. Stored on every entity as ``name_key`` and matched by
    indexed equality, which avoids the per-entity scan a ``toLower(name)``
    predicate forces.
    """
    return normalize_name(name).replace(" ", "")
