# GraphRAG SDK — Utils: Cypher Sanitization
# Prevents Cypher injection via label/type interpolation.

import json
from typing import Any


def sanitize_cypher_label(label: str) -> str:
    """Strip backticks to prevent Cypher injection in label/type positions.

    Labels and relationship types are interpolated into Cypher queries
    inside backtick-quoted identifiers (e.g. ``:`Label```). A malicious
    label containing backticks could break out of the identifier and
    inject arbitrary Cypher.

    Args:
        label: The raw label or relationship type string.

    Returns:
        Cleaned label safe for backtick-quoted interpolation.

    Raises:
        ValueError: If the label is empty or whitespace-only after cleaning.
    """
    cleaned = label.strip().replace("`", "")
    if not cleaned:
        raise ValueError(f"Invalid Cypher label: {label!r}")
    return cleaned


def merge_key_value(value: Any) -> Any:
    """A MERGE key value, compared the way FalkorDB compares it.

    Numbers are equal by value (``5`` and ``5.0`` are one key, as are
    ``0`` and ``-0.0``), a boolean is never equal to a number, and a string
    is never equal to either. Python's ``==`` and ``hash`` already treat
    numbers that way, so tagging the kind keeps ``True`` apart from ``1``
    and ``"5"`` apart from ``5`` without splitting ``5`` from ``5.0``.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    try:
        return ("other", json.dumps(value, sort_keys=True, default=str))
    except TypeError:
        # sort_keys cannot order mixed key types such as {"a": 1, 1: "b"}.
        return ("other", json.dumps(value, default=str))
