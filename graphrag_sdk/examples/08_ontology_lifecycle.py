"""
GraphRAG SDK -- Ontology lifecycle
==============================================
End-to-end walkthrough of the ontology API renamed in v1.2:

1. Declare an :py:class:`Ontology` with typed :py:class:`Attribute` properties.
2. Construct ``GraphRAG`` with ``ontology=...``.
3. Ingest text. The ontology is registered into the paired
   ``<graph>__ontology`` graph automatically.
4. Read the persisted ontology back with ``rag.get_ontology()``.
5. Round-trip the ontology through a JSON file (schema-as-config).
6. Show schema evolution rules:
   - Adding a new entity type on a subsequent ingest is OK.
   - Adding a new attribute to an existing type → ``OntologyModificationNotAllowedError``.
   - Re-declaring an existing attribute with a different type →
     ``OntologyContradictionError``.
7. Show that legacy names (``GraphSchema``, ``EntityType``, ``schema=``,
   ``rag.schema``) still work with a ``DeprecationWarning``.

Prerequisites:
    pip install graphrag-sdk[litellm]
    docker run -p 6379:6379 falkordb/falkordb
    export OPENAI_API_KEY=...

Usage:
    python 08_ontology_lifecycle.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import warnings
from pathlib import Path
from uuid import uuid4

from graphrag_sdk import (
    Attribute,
    ConnectionConfig,
    Entity,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
    Ontology,
    OntologyContradictionError,
    OntologyModificationNotAllowedError,
    Relation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")

# Demo text — small enough to extract quickly.
SAMPLE_TEXT_PASS_1 = """\
Marie Curie was born in 1867 in Warsaw. She worked at the Sorbonne in Paris.
Pierre Curie collaborated with her on radioactivity research.
"""

SAMPLE_TEXT_PASS_2 = """\
Albert Einstein was born in 1879 in Ulm. He worked at the Patent Office in Bern.
He later joined Princeton University, located in New Jersey.
"""


def build_ontology() -> Ontology:
    """Declare a small ontology with typed attributes.

    Entities (Person, Organization, Place) carry typed properties beyond the
    SDK-supplied ``name`` / ``description`` — the LLM is prompted to extract
    them, the values are coerced to the declared types at storage, and the
    Cypher-generation prompt surfaces them at retrieval time.
    """
    return Ontology(
        entities=[
            Entity(
                label="Person",
                description="A human being",
                properties=[
                    Attribute(name="birth_year", type="INTEGER"),
                    Attribute(name="birth_place", type="STRING"),
                ],
            ),
            Entity(label="Organization", description="A company or institution"),
            Entity(label="Place", description="A geographic location"),
        ],
        relations=[
            Relation(
                label="WORKS_AT",
                description="Employment",
                patterns=[("Person", "Organization")],
            ),
            Relation(
                label="LOCATED_IN",
                description="Physical containment",
                patterns=[("Organization", "Place")],
            ),
            Relation(
                label="COLLABORATED_WITH",
                description="Worked together on research",
                patterns=[("Person", "Person")],
            ),
        ],
    )


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    graph_name = f"ontology_demo_{uuid4().hex[:8]}"
    print(f"Using ephemeral graph: {graph_name}")

    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", port=6379, graph_name=graph_name),
        llm=LiteLLM(model="gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="text-embedding-3-large"),
        embedding_dimension=256,
        ontology=build_ontology(),
    )

    try:
        # ── 1. Ingest with the declared ontology ────────────────────────
        banner("1. Ingest text with the declared ontology")
        await rag.ingest(text=SAMPLE_TEXT_PASS_1)
        print("Ingested SAMPLE_TEXT_PASS_1.")

        # ── 2. Read back the persisted ontology ─────────────────────────
        banner("2. Inspect the persisted ontology")
        persisted = await rag.get_ontology()
        for e in persisted.entities:
            props = ", ".join(f"{p.name}:{p.type}" for p in e.properties) or "—"
            print(f"  Entity  {e.label:<14} properties: {props}")
        for r in persisted.relations:
            pats = ", ".join(f"{s}->{t}" for s, t in r.patterns) or "(any)"
            print(f"  Relation {r.label:<22} patterns: {pats}")

        # ── 3. Round-trip the ontology through a JSON file ──────────────
        banner("3. Save / load the ontology as JSON (schema-as-config)")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ontology.json"
            await rag.save_ontology(str(path))
            print(f"  Wrote: {path} ({path.stat().st_size} bytes)")
            loaded = Ontology.from_file(str(path))
            print(f"  Reloaded {len(loaded.entities)} entities, {len(loaded.relations)} relations.")

        # ── 4. Adding a new entity type on a fresh ingest is OK ─────────
        banner("4. Adding a new entity type via ingest (additive)")
        rag.ontology = Ontology(
            entities=list(persisted.entities)
            + [
                Entity(label="University", description="A higher-education institution"),
            ],
            relations=list(persisted.relations),
        )
        # Force re-init so the new schema is registered.
        rag._ontology_initialized = False
        await rag.ingest(text=SAMPLE_TEXT_PASS_2)
        after = await rag.get_ontology()
        assert any(e.label == "University" for e in after.entities), (
            "University should have been registered"
        )
        print("  University added to the ontology graph ✓")

        # ── 5. Adding a NEW attribute to an existing entity is REJECTED ─
        banner("5. Modifying an existing entity → OntologyModificationNotAllowedError")
        rag.ontology = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[
                        Attribute(name="birth_year", type="INTEGER"),
                        Attribute(name="birth_place", type="STRING"),
                        Attribute(name="nobel_year", type="INTEGER"),  # ← new
                    ],
                ),
            ],
        )
        rag._ontology_initialized = False
        try:
            await rag.ingest(text="dummy")
            raise AssertionError("expected OntologyModificationNotAllowedError")
        except OntologyModificationNotAllowedError as exc:
            print(f"  raised as expected: {exc}")

        # ── 6. Changing the TYPE of an existing attribute is REJECTED ──
        banner("6. Re-typing an existing attribute → OntologyContradictionError")
        rag.ontology = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[
                        Attribute(name="birth_year", type="STRING"),  # was INTEGER
                    ],
                ),
            ],
        )
        rag._ontology_initialized = False
        try:
            await rag.ingest(text="dummy")
            raise AssertionError("expected OntologyContradictionError")
        except OntologyContradictionError as exc:
            print(f"  raised as expected: {exc}")

        # ── 7. Re-declaring a subset is fine (treated as "use persisted")
        banner("7. Re-declaring an existing entity as a subset is OK")
        rag.ontology = Ontology(entities=[Entity(label="Person")])
        rag._ontology_initialized = False
        await rag._ensure_ontology_initialized()
        print("  Bare Person re-declaration accepted ✓")

        # ── 8. Legacy aliases still work (DeprecationWarning) ───────────
        banner("8. Legacy schema=... and class names still work (DeprecationWarning)")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Old code path: GraphSchema, EntityType, schema= kwarg.
            from graphrag_sdk import EntityType, GraphSchema

            legacy_rag = GraphRAG(
                connection=ConnectionConfig(
                    host="localhost", port=6379, graph_name=f"legacy_{uuid4().hex[:6]}"
                ),
                llm=LiteLLM(model="gpt-4o-mini"),
                embedder=LiteLLMEmbedder(model="text-embedding-3-large"),
                embedding_dimension=256,
                schema=GraphSchema(entities=[EntityType(label="Animal")]),
            )
            print(f"  legacy_rag.ontology = {[e.label for e in legacy_rag.ontology.entities]}")
            for w in caught:
                if issubclass(w.category, DeprecationWarning):
                    print(f"  ⚠️ DeprecationWarning: {w.message}")
            await legacy_rag.delete_all()
            await legacy_rag.close()

        print()
        print("All scenarios passed.")
    finally:
        # Clean up: drop both data + ontology graphs.
        try:
            await rag.delete_all()
        finally:
            await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
