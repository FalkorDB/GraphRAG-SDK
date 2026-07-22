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
# Wording kept close to the long-standing extraction instruction so that making
# it a shared constant does not change extraction behaviour (a reworded version
# shifted real-LLM output and destabilised integration tests).
ENTITY_DESCRIPTION_RULE = (
    "A concise 1-2 sentence description capturing the entity's key attributes "
    "and roles. It is embedded for semantic search, so it must be self-contained "
    "and understandable without the original text."
)
