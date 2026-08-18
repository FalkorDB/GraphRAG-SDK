"""
GraphRAG SDK -- Structured Ingestion
=======================================
Tables and documents in one graph, end to end:

  - declare a mapping          — which columns are entities, keys, properties
  - rag.ingest(csv, mapping=)  — deterministic, no model involved
  - rag.ingest(text=...)       — the ordinary extraction path, unchanged
  - rag.finalize()             — where the two halves resolve into one node
  - retrieval                  — including aggregation over typed columns

The problem this solves: run a CSV through the prose path and every cell
becomes a described entity, so ``age`` is the string "34" if it survives at
all. Nothing can be averaged, filtered numerically, or joined on a key. A
mapping fixes that by declaring identity and types up front, which also puts
them in the ontology where generated Cypher can see them.

The interesting part is not either half on its own. It is that "who works at
the company that reported the revenue miss, and how old are they" needs both:
the employment edges and ages come from the CSV, the revenue miss comes from
the note, and the answer only exists if they share one Acme Corp node.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]
    export OPENAI_API_KEY="sk-..."
"""

import asyncio
import csv
import tempfile
from pathlib import Path

from graphrag_sdk import (
    Column,
    ConnectionConfig,
    EdgeMapping,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
    MultiPathRetrieval,
    NodeMapping,
    RecordMapping,
    Table,
)

# ── The tables ──────────────────────────────────────────────────

ORGS_ROWS = [
    {"org_id": "ORG-42", "org_name": "Acme Corp", "hq_country": "US", "employee_count": "1200"},
    {"org_id": "ORG-7", "org_name": "Globex", "hq_country": "GB", "employee_count": "340"},
]

EMPLOYEE_ROWS = [
    {
        "employee_id": "E-1",
        "full_name": "Alice Smith",
        "age": "34",
        "job_title": "Engineer",
        "start_date": "2019-04-01",
        "org_id": "ORG-42",
    },
    {
        "employee_id": "E-2",
        "full_name": "Bob Jones",
        "age": "45",
        "job_title": "CFO",
        "start_date": "2015-11-01",
        "org_id": "ORG-42",
    },
    {
        "employee_id": "E-3",
        "full_name": "Carol White",
        "age": "29",
        "job_title": "Engineer",
        "start_date": "2021-02-15",
        "org_id": "ORG-7",
    },
]

BOARD_NOTE = (
    "Acme Corp reported a Q3 revenue miss, attributing the shortfall to delayed "
    "enterprise renewals. Alice Smith, an engineer at Acme Corp, presented the "
    "remediation plan. Bob Jones, the chief financial officer, said the shortfall "
    "would not affect the supply agreement with Globex."
)

# ── The mappings ────────────────────────────────────────────────

# One record is one organization. Table() is the shorthand for that case.
#
#   key   the column that identifies the entity. Its value becomes the node id,
#         so re-ingesting updates in place instead of duplicating.
#   name  the display name. A source holding both key and name also publishes
#         the id an extractor would compute for the same thing, which is what
#         lets this node and a node extracted from prose become one.
ORGS = Table(
    "Organization",
    key="org_id",
    name="org_name",
    hq_country="hq_country",
    employee_count=Column("employee_count", "INTEGER"),
)

# One record is a person, plus an edge to the organization it points at.
EMPLOYEES = RecordMapping(
    nodes=[
        NodeMapping(
            label="Person",
            key="employee_id",
            name="full_name",
            properties={
                "age": Column("age", "INTEGER"),
                "title": Column("job_title"),
                "start_date": Column("start_date", "DATE"),
            },
        ),
        # A foreign key: this row says the organization exists and gives its key,
        # but knows nothing else about it. reference=True writes it ON CREATE
        # only, so a pointer can never overwrite the name orgs.csv supplied.
        NodeMapping(label="Organization", key="org_id", reference=True),
    ],
    edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
)


def write_csv(directory: Path, name: str, rows: list[dict]) -> str:
    path = directory / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


async def main():
    llm = LiteLLM(model="openai/gpt-5.5")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256)
    workdir = Path(tempfile.mkdtemp())

    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="structured_demo"),
        llm=llm,
        embedder=embedder,
        embedding_dimension=256,
    ) as rag:
        # ── 1. The structured half ─────────────────────────────────
        # No model is involved. Same input, same graph, every time.
        orgs_csv = write_csv(workdir, "orgs.csv", ORGS_ROWS)
        employees_csv = write_csv(workdir, "employees.csv", EMPLOYEE_ROWS)

        print("── structured")
        for source, mapping in ((orgs_csv, ORGS), (employees_csv, EMPLOYEES)):
            result = await rag.ingest(source, mapping=mapping)
            print(f"   {Path(source).name}: {result}")

        # Every record is a Chunk, so a row is retrievable and traceable back to
        # its source exactly like a paragraph is. The typed projection lives on
        # the entity, where aggregation reads it.

        # ── 2. The unstructured half, unchanged ───────────────────
        print("── unstructured")
        result = await rag.ingest(text=BOARD_NOTE, document_id="board_note.txt")
        print(f"   board_note.txt: {result.nodes_created} nodes")

        # ── 3. Resolve ────────────────────────────────────────────
        # The note's "Acme Corp" and the CSV's ORG-42 are two nodes until here.
        # They agree on the name, so the ordinary resolver folds them together
        # and carries the typed columns onto the survivor.
        summary = await rag.finalize()
        print(f"── finalize: merged {summary.entities_deduplicated} duplicates")

        # ── 4. The ontology the mappings declared ─────────────────
        # This is what makes the columns queryable: generated Cypher can now see
        # that age is an INTEGER rather than guessing it is a described entity.
        ontology = await rag.get_ontology()
        for entity in ontology.entities:
            if entity.properties:
                declared = ", ".join(f"{p.name}:{p.type}" for p in entity.properties)
                print(f"   {entity.label}: {declared}")

        # ── 5. Retrieval ──────────────────────────────────────────
        # Aggregation needs the text-to-Cypher path, which needs the ontology.
        cypher_retrieval = MultiPathRetrieval(
            graph_store=rag._graph_store,
            vector_store=rag._vector_store,
            embedder=rag.embedder,
            llm=rag.llm,
            enable_cypher=True,
            ontology=ontology,
        )

        questions = [
            # Structured only: an average over a typed column.
            "What is the average age of employees at Acme Corp?",
            # Unstructured only.
            "Why did Acme Corp miss its revenue target?",
            # Both halves. Neither can answer it alone.
            "Who works at the company that reported the revenue miss, and how old are they?",
        ]
        print("── retrieval")
        for question in questions:
            answer = await rag.completion(question, strategy=cypher_retrieval)
            print(f"\n   Q: {question}\n   A: {answer.answer}")

        # ── 6. Updating a table ───────────────────────────────────
        # Node ids come from the declared key, so re-ingesting a corrected export
        # updates the same nodes. Nothing is duplicated and no id churns.
        EMPLOYEE_ROWS[0]["job_title"] = "Principal Engineer"
        await rag.ingest(write_csv(workdir, "employees.csv", EMPLOYEE_ROWS), mapping=EMPLOYEES)
        rows = await rag._graph_store.query_raw(
            "MATCH (p:Person {employee_id:'E-1'}) RETURN p.title, count(p)"
        )
        print(f"\n── after re-ingest: {rows.result_set}")


if __name__ == "__main__":
    asyncio.run(main())
