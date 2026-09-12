"""
GraphRAG SDK -- Structured Ingestion
=======================================
Tables and documents in one graph, end to end:

  - declare a mapping          — in the ontology, alongside the entity types
  - rag.ingest(csv)            — deterministic, no model involved
  - rag.ingest(text=...)       — the ordinary extraction path, unchanged
  - rag.finalize()             — embeddings, indexes, a resolver across sources
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
    Entity,
    GraphRAG,
    Link,
    LiteLLM,
    LiteLLMEmbedder,
    Ontology,
    TableMapping,
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

# ── The mappings, declared in the ontology ──────────────────────
# A mapping belongs to the schema, not to the call site: the labels and column
# types it declares have to be registered before any prose is extracted, or the
# extractor guesses a label and the table's rows can never join what it wrote.
# `source` is the filename `ingest` matches on, and it is also where each
# property's signature comes from — `age` from employees.csv is stored as
# `employees__age`, which is why no other source can overwrite it.

# One record is one organization: a key, a name, and two typed columns.
ORGS = TableMapping(
    source="orgs.csv",
    label="Organization",
    key="org_id",
    name="org_name",
    properties={
        "hq_country": Column("hq_country"),
        "employee_count": Column("employee_count", "INTEGER"),
    },
)

# One record is a person, plus a link to the organization it points at. The link
# is what turns an `org_id` column from text into an edge.
#
#   key    the column identifying the record, kept on the node as entity_key.
#          Links and a re-sync resolve through it, so re-ingesting a corrected
#          export updates the row in place.
#   name   the display name. The node id is derived from it exactly as it is
#          for a prose mention, which is what makes this node and one from
#          prose the same node from the first write.
#   links  a column pointing at another entity. The target is named only when
#          the pointer creates it, so it can never overwrite the name orgs.csv
#          supplied, and the two files can arrive in either order.
EMPLOYEES = TableMapping(
    source="employees.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={
        "age": Column("age", "INTEGER"),
        "title": Column("job_title"),
        "start_date": Column("start_date", "DATE"),
    },
    links=[Link("WORKS_AT", to="Organization", by="org_id")],
)

ONTOLOGY = Ontology(
    entities=[Entity(label="Person"), Entity(label="Organization")],
    tables=[ORGS, EMPLOYEES],
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
        ontology=ONTOLOGY,
        # The reason to declare column types at all. Without it, a question like
        # "what is the average age" has no passage to retrieve and cannot be
        # answered; with it, the question becomes a query against the ontology
        # the mappings declared.
        enable_cypher=True,
    ) as rag:
        # ── 1. The structured half ─────────────────────────────────
        # No model is involved. Same input, same graph, every time.
        orgs_csv = write_csv(workdir, "orgs.csv", ORGS_ROWS)
        employees_csv = write_csv(workdir, "employees.csv", EMPLOYEE_ROWS)

        print("── structured")
        for source in (orgs_csv, employees_csv):
            # No mapping argument: a .csv is records, and its declaration is
            # already in the ontology above.
            result = await rag.ingest(source)
            print(f"   {Path(source).name}: {result}")

        # Every record is a Chunk, so a row is retrievable and traceable back to
        # its source exactly like a paragraph is. The typed projection lives on
        # the entity, where aggregation reads it.

        # ── 2. The unstructured half, unchanged ───────────────────
        print("── unstructured")
        result = await rag.ingest(text=BOARD_NOTE, document_id="board_note.txt")
        print(f"   board_note.txt: {result.nodes_created} nodes")

        # ── 3. Finalize ───────────────────────────────────────────
        # The note's "Acme Corp" and orgs.csv's ORG-42 are already one node: a
        # row's id is derived from its name exactly as a mention's is, so there
        # was no merge to do. finalize() is where the graph becomes queryable as
        # a whole — entity and edge embeddings, indexes — and where a resolver
        # judges the names that do not match letter for letter across sources,
        # remembering a NO so the pair is never asked about again. Its report
        # lists what it could not decide for you.
        summary = await rag.finalize()
        print(
            f"── finalize: {summary.entities_deduplicated} merged, "
            f"{len(summary.property_conflicts)} property conflicts, "
            f"{len(summary.probable_duplicates)} probable duplicates to look at"
        )

        # ── 4. The ontology the mappings declared ─────────────────
        # This is what makes the columns queryable: generated Cypher can now see
        # that age is an INTEGER rather than guessing it is a described entity.
        ontology = await rag.get_ontology()
        for entity in ontology.entities:
            if entity.properties:
                declared = ", ".join(f"{p.name}:{p.type}" for p in entity.properties)
                print(f"   {entity.label}: {declared}")

        # ── 5. Retrieval ──────────────────────────────────────────
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
            answer = await rag.completion(question)
            print(f"\n   Q: {question}\n   A: {answer.answer}")

        # ── 6. Keeping the table in sync ──────────────────────────
        # A table is a snapshot, not an addition, so re-ingesting a source that
        # is already in the graph re-syncs it. Three things happen here at once:
        # Alice's title is corrected, a new hire appears, and Carol leaves the
        # export entirely.
        #
        # The last one is the case that needs the machinery. Rows are matched on
        # the declared key, so a changed row rewrites itself and a new row simply
        # arrives. A *removed* row has nothing left to rewrite it, so without a
        # re-sync it would sit in the graph forever.
        EMPLOYEE_ROWS[0]["job_title"] = "Principal Engineer"
        EMPLOYEE_ROWS.pop()  # Carol White is no longer in the export
        EMPLOYEE_ROWS.append(
            {
                "employee_id": "E-4",
                "full_name": "Dana Reed",
                "age": "41",
                "job_title": "COO",
                "start_date": "2020-06-01",
                "org_id": "ORG-42",
            }
        )
        result = await rag.ingest(write_csv(workdir, "employees.csv", EMPLOYEE_ROWS))
        print(f"\n── re-sync: {result}")

        for employee_id, name, title in await rag.query(
            "MATCH (p:Person) "
            "RETURN p.employees__employee_id, p.name, p.employees__title "
            "ORDER BY p.employees__employee_id"
        ):
            print(f"   {employee_id}  {name}  {title}")
        print("   Carol White is gone, Dana Reed is new, and E-1 kept its identity.")

        # And the note still owns what it legitimately knows, without either
        # source having to win. The export's title is `employees__title`; anything
        # the extractor decided from prose is unsigned and lands under its own
        # name. Both are on the node, so neither had to be dropped to make room
        # for the other.


if __name__ == "__main__":
    asyncio.run(main())
