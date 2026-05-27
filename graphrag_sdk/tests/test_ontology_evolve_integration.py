"""End-to-end integration tests for Groups 1+2 against real FalkorDB.

Gated by ``RUN_INTEGRATION=1``. Uses the ``real_falkordb_rag_factory``
fixture (see ``conftest.py``) so each test runs against its own
ephemeral graph and is cleaned up on teardown.
"""

from __future__ import annotations

import os

import pytest

from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation

from .conftest import _scripted_extraction_llm

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="Set RUN_INTEGRATION=1 to run real-FalkorDB integration tests",
)


@pytest.fixture
def starter_ontology() -> Ontology:
    return Ontology(
        entities=[
            Entity(
                label="Person",
                properties=[Attribute(name="age", type="INTEGER")],
            ),
            Entity(label="Company"),
        ],
        relations=[
            Relation(label="WORKS_AT", patterns=[("Person", "Company")]),
        ],
    )


@pytest.mark.asyncio
async def test_rename_entity_relabels_data_nodes(
    real_falkordb_rag_factory, starter_ontology
):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = _scripted_extraction_llm(
        [("Alice", "Person", "An engineer"), ("Acme", "Company", "A firm")]
    )
    rag = real_falkordb_rag_factory(
        llm=llm, resolver=ExactMatchResolution(), ontology=starter_ontology
    )
    await rag.ingest(text="Alice works at Acme.", document_id="doc1")

    new_ontology = await rag.rename_entity("Person", "Human")
    assert any(e.label == "Human" for e in new_ontology.entities)
    assert not any(e.label == "Person" for e in new_ontology.entities)

    stats = await rag.get_statistics()
    assert "Human" in stats["entity_types"]
    assert "Person" not in stats["entity_types"]


@pytest.mark.asyncio
async def test_drop_attribute_removes_property_and_declaration(
    real_falkordb_rag_factory, starter_ontology
):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = _scripted_extraction_llm([("Alice", "Person", "An engineer")])
    rag = real_falkordb_rag_factory(
        llm=llm, resolver=ExactMatchResolution(), ontology=starter_ontology
    )
    await rag.ingest(text="Alice is mentioned.", document_id="doc1")

    refreshed = await rag.drop_attribute("Person", "age")
    person = next(e for e in refreshed.entities if e.label == "Person")
    assert not any(a.name == "age" for a in person.properties)


@pytest.mark.asyncio
async def test_drop_entity_cascades_relations(
    real_falkordb_rag_factory, starter_ontology
):
    from graphrag_sdk.ingestion.resolution_strategies.exact_match import (
        ExactMatchResolution,
    )

    llm = _scripted_extraction_llm(
        [("Alice", "Person", "x"), ("Acme", "Company", "y")]
    )
    rag = real_falkordb_rag_factory(
        llm=llm, resolver=ExactMatchResolution(), ontology=starter_ontology
    )
    await rag.ingest(text="Alice works at Acme.", document_id="doc1")
    new_ontology = await rag.drop_entity("Company")
    assert not any(e.label == "Company" for e in new_ontology.entities)
    # The WORKS_AT pattern (Person, Company) is no longer declarable since
    # the Company entity is gone.
    assert all(
        ("Company" not in (s, t) for s, t in r.patterns)
        for r in new_ontology.relations
    )
