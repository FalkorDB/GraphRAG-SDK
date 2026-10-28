"""``finalize()`` gives the chunks it finds without a name a readable one.

A chunk is written without a ``name``, so a graph browser shows its UUID. The
step is a single Cypher ``SET``, which is what a mock cannot vouch for: the
string concatenation, ``toString`` on the index, the parameter carrying a
non-ASCII separator and the ``__pending__`` filter all have to be accepted by
a real server.

Requires ``RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import pytest

from graphrag_sdk import Column, Entity, ExactMatchResolution, Ontology, TableMapping


@pytest.fixture
def resolver():
    return ExactMatchResolution(resolve_property="name")


@pytest.fixture
def ontology():
    return Ontology(
        entities=[Entity(label="Person")],
        tables=[
            TableMapping(
                source="hr.csv",
                label="Person",
                key="employee_id",
                name="full_name",
                properties={"age": Column("age", "INTEGER")},
            )
        ],
    )


async def _load(rag, resolver, tmp_path):
    """One prose document and one table, so both ways of writing a chunk are covered."""
    (tmp_path / "hr.csv").write_text(
        "employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tim Cook,60\n"
    )
    await rag.ingest(str(tmp_path / "hr.csv"))
    await rag.ingest(
        text="Maya Ellison presented the roadmap.", document_id="note.txt", resolver=resolver
    )


async def _chunks(rag):
    """``(document id, index, name)`` for every chunk, in a stable order."""
    result = await rag._graph_store.query_raw(
        "MATCH (d:Document)-[:PART_OF]->(c:Chunk) RETURN d.id, c.index, c.name ORDER BY d.id, c.index"
    )
    return [tuple(row) for row in result.result_set]


class TestFinalizeNamesChunks:
    async def test_prose_and_record_chunks_get_readable_names(
        self, real_falkordb_rag_factory, llm, resolver, ontology, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(rag, resolver, tmp_path)
        before = await _chunks(rag)
        assert before, "the fixture documents wrote no chunks"
        assert all(name is None for _, _, name in before)

        result = await rag.finalize()

        after = await _chunks(rag)
        assert result.chunks_named == len(before)
        assert "note.txt" in {doc_id for doc_id, _, _ in after}
        for doc_id, index, name in after:
            assert name == f"{doc_id} · chunk {index}"

    async def test_a_second_finalize_names_nothing(
        self, real_falkordb_rag_factory, llm, resolver, ontology, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(rag, resolver, tmp_path)
        await rag.finalize()
        named = await _chunks(rag)

        again = await rag.finalize()

        assert again.chunks_named == 0
        assert await _chunks(rag) == named

    async def test_a_chunk_that_already_has_a_name_keeps_it(
        self, real_falkordb_rag_factory, llm, resolver, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver)
        await rag.ingest(
            text="Maya Ellison presented the roadmap.", document_id="note.txt", resolver=resolver
        )
        await rag._graph_store.query_raw("MATCH (c:Chunk {index: 0}) SET c.name = 'mine'")

        result = await rag.finalize()

        assert result.chunks_named == 0
        assert [name for _, _, name in await _chunks(rag)] == ["mine"]

    async def test_name_chunks_false_leaves_chunks_as_written(
        self, real_falkordb_rag_factory, llm, resolver, ontology, tmp_path
    ):
        rag = real_falkordb_rag_factory(llm=llm, resolver=resolver, ontology=ontology)
        await _load(rag, resolver, tmp_path)

        result = await rag.finalize(name_chunks=False)

        assert result.chunks_named == 0
        assert all(name is None for _, _, name in await _chunks(rag))
