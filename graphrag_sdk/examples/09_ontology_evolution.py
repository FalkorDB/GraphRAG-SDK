"""
GraphRAG SDK — Ontology evolution walkthrough
==============================================
Companion to ``08_ontology_lifecycle.py`` (which demonstrates strict
additive evolution). This script shows the *mutating* evolution API:

The design enforces a strict alignment invariant: **a declared attribute
exists on every instance of its owner entity type**. To honor this,
``add_attribute`` is atomic — it LLM-extracts values from existing
chunks before committing the schema change.

1. Ingest a small corpus with a starter ontology.
2. Group 1 — pure declarations: ``add_entity``, ``add_relation_pattern``.
3. Group 2 — mechanical migration: ``rename_entity``, ``rename_attribute``.
4. Atomic attribute evolution: ``add_attribute`` (LLM backfill + commit
   in one call), ``drop_attribute`` (cheap data + ontology removal).
5. Type changes: ``drop_attribute`` + ``add_attribute`` with the new
   type — the LLM re-derives values from the chunks.
6. Opportunistic discovery: ``backfill_entity``, ``backfill_relation_pattern``.

Prereqs::

    pip install graphrag-sdk[litellm]
    docker run -p 6379:6379 falkordb/falkordb
    export OPENAI_API_KEY=...

Run::

    python 09_ontology_evolution.py

Important: do NOT run ``ingest()`` concurrently with ``add_attribute()``
or ``drop_attribute()``. The extractor reads the persisted ontology to
decide what to extract; concurrent ingest during attribute evolution
would create entities without the new attribute. Coordinate at the
application level.
"""

from __future__ import annotations

import asyncio
import logging
import os
from uuid import uuid4

from graphrag_sdk import (
    Attribute,
    ConnectionConfig,
    Entity,
    EvolutionResult,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
    Ontology,
    Relation,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")


SAMPLE_TEXT = """\
Alice Liddell is a software engineer at Acme Corporation. She joined in 2020.
Bob Carroll works at Acme Corporation as a product manager.
Acme Corporation is based in Boston.
"""


def starter_ontology() -> Ontology:
    return Ontology(
        entities=[
            Entity(label="Person", description="A human"),
            Entity(label="Company", description="A business"),
        ],
        relations=[
            Relation(
                label="WORKS_AT",
                description="Employment",
                patterns=[("Person", "Company")],
            ),
        ],
    )


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_evolution(label: str, result: EvolutionResult) -> None:
    print(
        f"  {label}: filled={result.values_filled} "
        f"chunks_scanned={result.chunks_scanned} "
        f"chunks_skipped={result.chunks_skipped} "
        f"llm_calls={result.llm_calls} "
        f"elapsed={result.elapsed_s:.2f}s"
    )


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    graph_name = f"evolve_{uuid4().hex[:8]}"
    print(f"Using ephemeral graph: {graph_name}")

    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", port=6379, graph_name=graph_name),
        llm=LiteLLM(model="gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="text-embedding-3-large"),
        embedding_dimension=256,
        ontology=starter_ontology(),
    )

    try:
        # ── 1. Ingest with the starter ontology ────────────────────
        banner("1. Ingest")
        await rag.ingest(text=SAMPLE_TEXT, document_id="evolution-demo")

        # ── 2. Atomic add_attribute (declare + LLM backfill + commit) ─
        banner("2. add_attribute('Person', 'role') — atomic evolve+backfill")
        result = await rag.add_attribute("Person", Attribute(name="role", type="STRING"))
        print_evolution("Person.role", result)
        # On return, the ontology graph has the new property AND every
        # mentioned Person has been LLM-asked for their role (null where
        # the LLM honestly doesn't know).

        # ── 3. Add another attribute (independent atomic backfill) ─
        # Re-issuing the same add_attribute would raise ValueError now
        # because `role` is already declared. This is a fresh, independent
        # backfill operation — different op_id (Person.join_year), so it
        # gets its own chunk markers and runs its own LLM scan.
        banner("3. Add another attribute — independent atomic backfill")
        result = await rag.add_attribute("Person", Attribute(name="join_year", type="INTEGER"))
        print_evolution("Person.join_year", result)

        # ── 4. Type change: drop + add (LLM re-derives) ────────────
        banner("4. Change Person.join_year STRING via drop + add")
        await rag.drop_attribute("Person", "join_year")
        result = await rag.add_attribute("Person", Attribute(name="join_year", type="STRING"))
        print_evolution("Person.join_year retyped", result)

        # ── 5. Group 1 / 2 operations (cheap, no LLM) ──────────────
        banner("5. Cheap structural changes")
        await rag.rename_entity("Person", "Employee")
        await rag.rename_attribute("Employee", "role", "title")
        ontology = await rag.get_ontology()
        labels = [e.label for e in ontology.entities]
        print(f"  Entity labels: {labels}")
        emp = next(e for e in ontology.entities if e.label == "Employee")
        print(f"  Employee attrs: {[(a.name, a.type) for a in emp.properties]}")

        # ── 6. Opportunistic discovery (opt-in, not invariant-enforcing) ─
        banner("6. backfill_entity — opportunistic: scan corpus for missed Cities")
        await rag.add_entity(Entity(label="City", description="A geographic place"))
        bf = await rag.backfill_entity("City", scope="all")
        print(f"  City entities found via opportunistic scan: {bf.values_filled}")

        banner("7. backfill_relation_pattern — opportunistic edge discovery")
        await rag.add_relation_pattern("BASED_IN", "Company", "City")
        bf = await rag.backfill_relation_pattern("BASED_IN", "Company", "City")
        print(f"  BASED_IN edges added: {bf.values_filled}")

        # ── 8. Inspect final ontology ─────────────────────────────
        banner("8. Final ontology")
        ontology = await rag.get_ontology()
        for e in ontology.entities:
            props = ", ".join(f"{p.name}:{p.type}" for p in e.properties) or "—"
            print(f"  Entity   {e.label:<14} props: {props}")
        for r in ontology.relations:
            pats = ", ".join(f"{s}->{t}" for s, t in r.patterns) or "(open)"
            print(f"  Relation {r.label:<18} patterns: {pats}")

        print()
        print("All scenarios passed.")
    finally:
        try:
            await rag.delete_all()
        finally:
            await rag.close()


if __name__ == "__main__":
    asyncio.run(main())
