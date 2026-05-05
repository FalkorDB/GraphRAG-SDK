"""
GraphRAG SDK -- Incremental Updates
======================================
Demonstrates the v1.1.0 incremental ingestion primitives:

  - rag.update(...)             — re-sync a changed document
  - rag.delete_document(...)    — remove one document and its orphans
  - rag.apply_changes(...)      — heterogeneous batch (added/modified/deleted)

The canonical use case is CI-driven graph updates on PR merge: a typical
PR diff has all three change types in one go, and apply_changes routes
each list to the right primitive so consumer code stays flat.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]
    export OPENAI_API_KEY="sk-..."
"""

import asyncio

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)


V1_TEXT = (
    "Alice Johnson is a software engineer at Acme Corp in London. "
    "She reports to Bob Smith."
)
V2_TEXT = (
    "Alice Johnson is the VP of Engineering at Acme Corp in Berlin. "
    "She reports to the CEO, Carol Wei."
)
NEW_DOC = (
    "Acme Corp acquired Globex Industries in 2026. "
    "The combined company employs 5000 people."
)


async def main():
    llm = LiteLLM(model="openai/gpt-5.5")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256)

    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="incremental_demo"),
        llm=llm,
        embedder=embedder,
        embedding_dimension=256,
    ) as rag:
        # ── 1. Initial ingest ──────────────────────────────────────
        # Pass a stable document_id so update() / delete_document()
        # can target the same node later. (In file mode you can omit
        # document_id and the path is used by default.)
        await rag.ingest(text=V1_TEXT, document_id="alice_bio")
        print("Initial state ingested.")

        # ── 2. Touch-only update — no-op short-circuit ─────────────
        # The content_hash on the Document node matches, so this is
        # effectively a single Cypher lookup.
        result = await rag.update(text=V1_TEXT, document_id="alice_bio")
        assert result.no_op is True
        print(f"No-op update detected (content hash matched).")

        # ── 3. Real update — chunks replaced, orphans cleaned up ───
        # Bob Smith was only mentioned by V1; he becomes an orphan
        # entity when V2 replaces V1, and update() removes him along
        # with his RELATES edges.
        result = await rag.update(text=V2_TEXT, document_id="alice_bio")
        print(
            f"Updated alice_bio: replaced {result.chunks_deleted} chunks, "
            f"removed {result.entities_deleted} orphan entities, "
            f"wrote {result.chunks_indexed} new chunks."
        )

        # ── 4. Add a second document ───────────────────────────────
        await rag.ingest(text=NEW_DOC, document_id="acquisitions_2026")
        print("Added a second document.")

        # ── 5. Heterogeneous batch via apply_changes ───────────────
        # Imagine a PR diff: alice_bio renamed elsewhere (delete),
        # acquisitions_2026 changed (modified), one new doc (added).
        batch = await rag.apply_changes(
            added=[],  # new file paths would go here in file mode
            modified=[],  # typically file paths whose content changed
            deleted=["alice_bio"],
        )
        delete_succeeded = sum(
            1 for r in batch.deleted if not isinstance(r, Exception)
        )
        print(
            f"apply_changes: deleted {delete_succeeded}, "
            f"modified {len(batch.modified)}, "
            f"added {len(batch.added)}"
        )

        # ── 6. Finalize ONCE at the end of the batch ───────────────
        # finalize() is O(graph size) for cross-document dedup, so
        # call it after the whole batch — never per file. apply_changes
        # deliberately does not call finalize internally.
        finalize_result = await rag.finalize()
        print(
            f"finalize: deduplicated {finalize_result.entities_deduplicated} "
            f"entities, embedded {finalize_result.entities_embedded} entities"
        )

        # ── 7. Verify with a query ─────────────────────────────────
        answer = await rag.completion("Who acquired Globex Industries?")
        print(f"\nQ: Who acquired Globex Industries?")
        print(f"A: {answer.answer}")


if __name__ == "__main__":
    asyncio.run(main())
