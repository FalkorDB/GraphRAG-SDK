# GraphRAG SDK — Utils: Cypher Sanitization
# Prevents Cypher injection via label/type interpolation.


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
