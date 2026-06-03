"""End-to-end attribute-evolution tests against a real LLM + FalkorDB.

Gated by ``RUN_INTEGRATION=1`` AND ``OPENAI_API_KEY``. The LLM call cost
is small (one focused prompt per chunk) but real — keep the test corpus
tiny.
"""

from __future__ import annotations

import os

import pytest

from graphrag_sdk.core.models import Attribute, Entity, Ontology

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="Requires RUN_INTEGRATION=1 and OPENAI_API_KEY",
)


async def test_add_attribute_fills_values_from_chunks(
    real_falkordb_rag_factory,
):
    """add_attribute is atomic: declare + LLM backfill + commit. After it
    returns, every Person mentioned in chunks has been LLM-asked for its
    age value (or null where the LLM honestly doesn't know)."""
    from graphrag_sdk.core.providers import LiteLLM
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = LiteLLM(model="gpt-4o-mini")
    starter = Ontology(entities=[Entity(label="Person")])
    rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution(), ontology=starter)
    await rag.ingest(
        text="Alice is 32 years old. Bob is 27. They are both engineers.",
        document_id="people",
    )

    # Atomic: schema declaration + LLM backfill + commit, in one call.
    result = await rag.add_attribute("Person", Attribute(name="age", type="INTEGER"))
    assert result.values_filled >= 1, "expected at least one age value filled"

    # The ontology graph has been updated as the commit point.
    ontology = await rag.get_ontology()
    person = next(e for e in ontology.entities if e.label == "Person")
    assert any(a.name == "age" and a.type == "INTEGER" for a in person.properties)


async def test_add_attribute_rejects_duplicate(real_falkordb_rag_factory):
    """Re-declaring the same attribute raises — type changes go through
    drop_attribute + add_attribute with the new type."""
    from graphrag_sdk.core.providers import LiteLLM
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = LiteLLM(model="gpt-4o-mini")
    starter = Ontology(entities=[Entity(label="Person")])
    rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution(), ontology=starter)
    await rag.ingest(text="Alice is 32 years old.", document_id="people")

    await rag.add_attribute("Person", Attribute(name="age", type="INTEGER"))
    with pytest.raises(ValueError, match="already declared"):
        await rag.add_attribute("Person", Attribute(name="age", type="STRING"))


async def test_drop_then_add_triggers_fresh_backfill(real_falkordb_rag_factory):
    """op_id includes the attribute type, so drop+add with a new type
    runs a NEW backfill (chunk markers from the previous type don't
    short-circuit it). Without this, the documented type-change pattern
    would silently commit the new schema with stale values."""
    from graphrag_sdk.core.providers import LiteLLM
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = LiteLLM(model="gpt-4o-mini")
    starter = Ontology(entities=[Entity(label="Person")])
    rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution(), ontology=starter)
    await rag.ingest(text="Alice is 32 years old.", document_id="people")

    # First add as INTEGER.
    await rag.add_attribute("Person", Attribute(name="age", type="INTEGER"))

    # Type change: drop then add with new type.
    await rag.drop_attribute("Person", "age")
    second = await rag.add_attribute("Person", Attribute(name="age", type="STRING"))

    # The second add MUST have made fresh LLM calls — its op_id differs
    # from the INTEGER one, so chunk markers don't filter it out.
    assert second.llm_calls >= 1, (
        "drop+add with new type must trigger a fresh backfill — chunk "
        "markers from the previous (INTEGER) op_id must not short-circuit it."
    )
