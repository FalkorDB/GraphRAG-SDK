"""
GraphRAG SDK — Ontology discovery walkthrough
=============================================
Bookend to the evolution example (``09_ontology_evolution.py``). That
script assumes you already have a curated starter ontology. This one
shows the *discovery* side: how to get one from a corpus, save it,
ingest with it, and later propose schema additions as new documents
arrive.

Two cooperating APIs:

  1. ``Ontology.from_sources(...)`` — pure function on the data model.
     Bootstraps a draft from raw documents. No DB connection needed.
     Run it, inspect the draft, save it to JSON, then hand it to
     ``GraphRAG``.

  2. ``GraphRAG.suggest_schema_extensions(...)`` — live-graph proposer.
     Once you have a committed ontology and you're ingesting more docs,
     this scans new sources, uses the committed ontology as a soft
     controlled vocabulary, and returns a ``SchemaExtensionProposal``
     with additions only. Nothing is applied — you review and apply via
     ``add_entity`` / ``add_relation_pattern`` / ``add_attribute``.

Prereqs::

    pip install graphrag-sdk[litellm]
    docker run -p 6379:6379 falkordb/falkordb
    export OPENAI_API_KEY=...

Run::

    python 10_ontology_discovery.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
    Ontology,
    SchemaExtensionProposal,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s")


# Two short documents for the bootstrap corpus and one extra for the
# extension-proposal demo. Real corpora would be larger; these are
# tiny enough to run cheaply.

BOOTSTRAP_DOC_A = """\
Alice Liddell is a software engineer at Acme Corporation. She joined Acme in 2020.
Acme Corporation is a technology company based in Boston, Massachusetts.
"""

BOOTSTRAP_DOC_B = """\
Bob Carroll works at Globex Industries as a product manager.
Globex Industries is a manufacturing company headquartered in San Francisco.
"""

EXTENSION_DOC = """\
Carol White is the CEO of Initech Inc.
Initech Inc. acquired Acme Corporation in 2023 for $500 million.
The acquisition was announced at a press conference in New York.
"""


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_ontology(ontology: Ontology) -> None:
    for e in ontology.entities:
        props = ", ".join(f"{p.name}:{p.type}" for p in e.properties) or "—"
        print(f"  Entity   {e.label:<18} props: {props}")
    for r in ontology.relations:
        pats = ", ".join(f"{s}->{t}" for s, t in r.patterns) or "(open)"
        print(f"  Relation {r.label:<18} patterns: {pats}")


def print_proposal(proposal: SchemaExtensionProposal) -> None:
    print(f"  {proposal.summary()}")
    for e in proposal.new_entities:
        props = ", ".join(p.name for p in e.properties) or "—"
        print(f"    + Entity   {e.label} (props: {props})")
    for r in proposal.new_relations:
        pats = ", ".join(f"{s}->{t}" for s, t in r.patterns) or "(open)"
        print(f"    + Relation {r.label} (patterns: {pats})")
    for rel_label, src, tgt in proposal.new_patterns:
        print(f"    + Pattern  {rel_label}: {src}->{tgt}")
    for owner, attr in proposal.new_attributes:
        print(f"    + Attr     {owner}.{attr.name} ({attr.type})")


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    llm = LiteLLM(model="gpt-4o-mini")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        doc_a = tmp / "doc_a.md"
        doc_b = tmp / "doc_b.md"
        doc_extra = tmp / "doc_extra.md"
        doc_a.write_text(BOOTSTRAP_DOC_A)
        doc_b.write_text(BOOTSTRAP_DOC_B)
        doc_extra.write_text(EXTENSION_DOC)

        # ── 1. Bootstrap an ontology draft from raw documents ─────
        banner("1. Ontology.from_sources — bootstrap from corpus")
        draft = await Ontology.from_sources(
            [str(doc_a), str(doc_b)],
            llm,
            boundaries="business news: companies, employees, locations",
            sample_chunks_per_doc=2,
        )
        print_ontology(draft)

        # ── 2. Persist the draft & re-load ─────────────────────────
        banner("2. Save / reload (schema-as-config)")
        schema_path = tmp / "ontology.json"
        draft.save_to_file(str(schema_path))
        print(f"  Wrote {schema_path}")
        reloaded = Ontology.from_file(str(schema_path))
        assert reloaded.model_dump() == draft.model_dump()
        print("  Round-trip OK.")

        # ── 3. Ingest with the discovered draft ───────────────────
        banner("3. Ingest with the discovered ontology")
        graph_name = f"discover_{uuid4().hex[:8]}"
        print(f"  Using ephemeral graph: {graph_name}")
        rag = GraphRAG(
            connection=ConnectionConfig(
                host="localhost", port=6379, graph_name=graph_name
            ),
            llm=llm,
            embedder=LiteLLMEmbedder(model="text-embedding-3-large"),
            embedding_dimension=256,
            ontology=reloaded,
        )
        try:
            await rag.ingest(source=[str(doc_a), str(doc_b)])
            print("  Ingested 2 documents.")

            # ── 4. Suggest schema extensions from a new doc ───────
            banner("4. suggest_schema_extensions — propose deltas from new doc")
            proposal = await rag.suggest_schema_extensions(
                str(doc_extra),
                boundaries="business news: companies, employees, locations",
                sample_chunks_per_doc=2,
            )
            print_proposal(proposal)

            # ── 5. Apply accepted parts via the v1.2.x mutation API ─
            if not proposal.is_empty:
                banner("5. Apply the proposal via add_* / add_attribute")
                for entity in proposal.new_entities:
                    await rag.add_entity(entity)
                    print(f"  + add_entity({entity.label})")
                for relation in proposal.new_relations:
                    for src, tgt in relation.patterns:
                        await rag.add_relation_pattern(relation.label, src, tgt)
                        print(f"  + add_relation_pattern({relation.label}, {src}, {tgt})")
                for rel_label, src, tgt in proposal.new_patterns:
                    await rag.add_relation_pattern(rel_label, src, tgt)
                    print(f"  + add_relation_pattern({rel_label}, {src}, {tgt})")
                for owner, attr in proposal.new_attributes:
                    # add_attribute is atomic + LLM-backfilled.
                    result = await rag.add_attribute(owner, attr)
                    print(
                        f"  + add_attribute({owner}, {attr.name}): "
                        f"filled={result.values_filled} llm_calls={result.llm_calls}"
                    )
            else:
                print("  Nothing to apply — the new doc fit the existing schema.")

            # ── 6. Inspect the final committed ontology ──────────
            banner("6. Final committed ontology")
            print_ontology(await rag.get_ontology())
        finally:
            try:
                await rag.delete_all()
            finally:
                await rag.close()

        print()
        print("Discovery walkthrough complete.")


if __name__ == "__main__":
    asyncio.run(main())
