"""End-to-end backfill tests against a real LLM + FalkorDB.

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


async def test_backfill_attribute_fills_missing_values(
    real_falkordb_rag_factory,
):
    """Ingest with no `age` attribute → add it → backfill → values appear."""
    from graphrag_sdk.core.providers import LiteLLM
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = LiteLLM(model="gpt-4o-mini")
    starter = Ontology(
        entities=[Entity(label="Person")],
    )
    rag = real_falkordb_rag_factory(llm=llm, resolver=ExactMatchResolution(), ontology=starter)
    await rag.ingest(
        text="Alice is 32 years old. Bob is 27. They are both engineers.",
        document_id="people",
    )

    # Add the new attribute (declaration only — no values yet).
    await rag.add_attribute("Person", Attribute(name="age", type="INTEGER"))

    # Backfill the attribute from existing chunks.
    result = await rag.backfill_attribute("Person", "age")
    assert result.values_filled >= 1, "expected at least one age value filled"

    # Re-running is idempotent — no new LLM calls because every chunk in
    # scope was already marked.
    result2 = await rag.backfill_attribute("Person", "age")
    assert result2.llm_calls == 0
