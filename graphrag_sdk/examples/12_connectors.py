"""
GraphRAG SDK -- Connectors
==========================
Keep the graph in step with a database table and a folder of documents:

  - sql_database(...)        — dlt's database source; rows become a table
  - file_documents(...)      — files from a dlt filesystem listing
  - sync(rag, [...])         — writes what changed, removes what is gone
  - sync(..., dry_run=True)  — what a sync would do, before any model call

Run it twice. The second sync makes no extraction calls and reports every
document unchanged (the question at the end still asks the model). Edit a note
or delete one, run it again, and only that note is re-extracted or removed.

The folder here is local so the example runs as is. The same filesystem
source reads S3 (s3://bucket/path), Google Cloud Storage (gs://) and Google
Drive (gdrive://<folder id>) with dlt's credentials. For Notion, Confluence
or SharePoint, wrap a LlamaIndex reader with llamaindex_documents().

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb:v4.22.2
    pip install graphrag-sdk[litellm,dlt] "dlt[sql_database]"
    export OPENAI_API_KEY="sk-..."
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from dlt.sources.filesystem import filesystem
from dlt.sources.sql_database import sql_database

from graphrag_sdk import (
    ConnectionConfig,
    Entity,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
    Ontology,
    TableMapping,
)
from graphrag_sdk.connectors import file_documents, sync

# Kept between runs, so a second run has something to compare against.
HERE = Path(tempfile.gettempdir()) / "graphrag_connectors_demo"


def prepare_sources() -> str:
    """A SQLite table of employees and a folder of notes about them."""
    notes = HERE / "notes"
    if not notes.exists():
        notes.mkdir(parents=True)
        for name, text in {
            "acme.md": "Acme Corp missed its Q3 revenue target. Alice leads the recovery plan.",
            "initech.md": "Bob moved from Acme Corp to Initech in September.",
        }.items():
            (notes / name).write_text(text)

    db = HERE / "crm.sqlite"
    initialize = not db.exists()
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT)")
        if initialize:
            con.execute("INSERT INTO employees VALUES (1, 'Alice'), (2, 'Bob')")
    return f"sqlite:///{db}"


async def main():
    database_url = prepare_sources()
    ontology = Ontology(
        entities=[Entity(label="Person"), Entity(label="Organization")],
        # The table's name, as the database calls it, finds its mapping.
        tables=[TableMapping(source="employees", label="Person", key="id", name="name")],
    )

    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="connectors_demo"),
        llm=LiteLLM(model="openai/gpt-5.5"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256),
        ontology=ontology,
    ) as rag:
        data = [
            sql_database(credentials=database_url).with_resources("employees"),
            file_documents(
                filesystem(bucket_url=(HERE / "notes").as_uri(), file_glob="*.md"),
                name="notes",
            ),
        ]

        preview = await sync(rag, data, dry_run=True)
        for report in preview.documents.values():
            print("would sync", report.summary())
        for table, rows in preview.table_rows.items():
            print(f"would sync {table}: {rows} rows")

        result = await sync(rag, data)
        for report in result.documents.values():
            print("synced", report.summary())
        for table, written in result.tables.items():
            print(f"synced {table}: {written.records} rows")

        await rag.finalize()
        answer = await rag.completion("Who leads the recovery plan at Acme Corp?")
        print(answer.answer)


if __name__ == "__main__":
    asyncio.run(main())
