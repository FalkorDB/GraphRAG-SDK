"""
GraphRAG SDK — Ontology evolution walkthrough
==============================================
Companion to ``08_ontology_lifecycle.py`` (which demonstrates strict
additive evolution). This script shows the *mutating* evolution API:

1. Ingest a small corpus with a starter ontology.
2. **Group 1** — pure ontology changes: ``add_attribute``, ``add_relation_pattern``.
3. **Group 2** — mechanical data migration: ``rename_entity``,
   ``drop_attribute``, ``retype_attribute``.
4. **Group 3** — LLM-driven backfill: ``backfill_attribute``,
   ``backfill_entity``, ``backfill_relation_pattern``.
5. Idempotency demo — re-run a backfill and observe ``llm_calls == 0``.

Prereqs::

    pip install graphrag-sdk[litellm]
    docker run -p 6379:6379 falkordb/falkordb
    export OPENAI_API_KEY=...

Run::

    python 09_ontology_evolution.py

The focused backfill prompts are intentionally narrower than the
production ``VERIFY_EXTRACT_RELS_PROMPT``. They have not been A/B-tested
on a representative corpus yet — expect to refine wording per your
domain.
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
        # ── 1. Ingest the corpus with the starter ontology ─────────
        banner("1. Ingest")
        await rag.ingest(text=SAMPLE_TEXT, document_id="evolution-demo")

        # ── 2. Group 1: declare new structure (no data change) ─────
        banner("2. Declare a new attribute on Person (no values yet)")
        await rag.add_attribute("Person", Attribute(name="role", type="STRING"))
        await rag.add_attribute("Person", Attribute(name="join_year", type="INTEGER"))
        ontology = await rag.get_ontology()
        person = next(e for e in ontology.entities if e.label == "Person")
        print("  Person attrs:", [(a.name, a.type) for a in person.properties])

        # ── 3. Group 2: mechanical data migration ─────────────────
        banner("3. Mechanical migration: rename Person → Employee, drop join_year")
        await rag.rename_entity("Person", "Employee")
        await rag.drop_attribute("Employee", "join_year")
        ontology = await rag.get_ontology()
        labels = [e.label for e in ontology.entities]
        print(f"  Entity labels after rename: {labels}")

        # ── 4. Group 3: LLM-driven backfill ───────────────────────
        banner("4. Backfill Employee.role from the existing chunks")
        result = await rag.backfill_attribute("Employee", "role")
        print(f"  op_id={result.operation_id}")
        print(f"  chunks_scanned={result.chunks_scanned} values_filled={result.values_filled}")
        print(f"  llm_calls={result.llm_calls} elapsed={result.elapsed_s:.2f}s")

        # ── 5. Idempotency ────────────────────────────────────────
        banner("5. Re-run the same backfill — should be a no-op (markers hit)")
        result2 = await rag.backfill_attribute("Employee", "role")
        print(f"  llm_calls={result2.llm_calls} (expected 0)")
        print(f"  chunks_skipped={result2.chunks_skipped}")

        # ── 6. Add a new entity type and backfill its instances ───
        banner("6. Declare a new entity type (City) and backfill instances")
        await rag.add_entity(Entity(label="City", description="A geographic place"))
        entity_result = await rag.backfill_entity("City", scope="all")
        print(
            f"  City entities found: {entity_result.values_filled} "
            f"across {entity_result.chunks_scanned} chunks"
        )

        # ── 7. Add a new relation pattern and backfill edges ──────
        banner("7. Declare a new relation pattern (BASED_IN) and backfill edges")
        await rag.add_relation_pattern("BASED_IN", "Company", "City")
        pattern_result = await rag.backfill_relation_pattern(
            "BASED_IN", "Company", "City"
        )
        print(
            f"  BASED_IN edges added: {pattern_result.values_filled} "
            f"across {pattern_result.chunks_scanned} chunks"
        )

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
