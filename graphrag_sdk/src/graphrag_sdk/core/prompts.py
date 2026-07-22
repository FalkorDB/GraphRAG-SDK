"""Shared prompt fragments.

Keeping cross-cutting instructions in one place so the steps that must agree —
notably how an entity *description* is written — cannot drift apart.
"""

# The single rule for how an entity description is produced. Used verbatim by
# entity extraction (writing each entity's description) and by resolution
# (writing the merged description when entities are unified), so both steps
# emit descriptions of the same shape and length. Because the rule bounds the
# text, callers can feed descriptions to the LLM in full — no ad-hoc truncation,
# which is what caused rich descriptions to erode across repeated merges.
ENTITY_DESCRIPTION_RULE = (
    "Describe the entity in a self-contained way — what it is and its key "
    "facts and roles — understandable without the source text (it is embedded "
    "for semantic search). Be concise and factual: at most 2-3 sentences, no "
    "filler, no restating the name."
)
