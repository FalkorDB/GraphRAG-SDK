---
title: "Ingest PDF and Markdown"
nav_order: 4
parent: "Guides"
grand_parent: "GraphRAG-SDK"
description: "Multi-format ingestion. PDF and Markdown use auto-detected loaders; supply a custom LoaderStrategy for anything else."
---

# Ingest PDF and Markdown

`ingest()` auto-selects a loader by file extension. The defaults cover plain text, Markdown, and PDF. For other formats, write a custom `LoaderStrategy` or extract the text yourself and call `ingest(text=...)`.

## Install the right extras

```bash
# Plain text + Markdown only — already included in the base install.
pip install graphrag-sdk[litellm]

# Add PDF support (pypdf-based, MIT-licensed).
pip install graphrag-sdk[litellm,pdf]

# Or PDF with table-aware extraction (PyMuPDF, AGPL-3.0).
pip install graphrag-sdk[litellm,pdf-fast]

# Or every extra at once.
pip install graphrag-sdk[all]
```

PyMuPDF is much better at tables but is AGPL-licensed — choose deliberately.

## Runnable example — mixed batch

```python
import asyncio
from graphrag_sdk import GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder


async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="multi_format"),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
    ) as rag:
        # Auto-detected per file:
        # - .pdf  -> PdfLoader
        # - .md   -> MarkdownLoader (parses headings into structural elements)
        # - .txt  -> TextLoader
        results = await rag.ingest(
            source=["docs/whitepaper.pdf", "docs/api.md", "README.txt"],
            max_concurrency=3,
        )

        for r in results:
            if isinstance(r, Exception):
                print(f"FAILED: {r}")
            else:
                print(f"{r.document_info.path}: "
                      f"{r.nodes_created} nodes, {r.relationships_created} edges")

        await rag.finalize()


asyncio.run(main())
```

`ingest()` in batch mode returns `list[IngestionResult | Exception]` — one slot per source, aligned by index. One bad file does not abort the batch; inspect each entry. Failures are also logged at WARNING.

## Override the loader explicitly

When auto-detection isn't what you want — for example, a file with a `.dat` extension that's actually Markdown — pass `loader=` directly:

```python
from graphrag_sdk.ingestion.loaders.markdown_loader import MarkdownLoader

await rag.ingest(
    source="data/notes.dat",
    loader=MarkdownLoader(),
)
```

## Custom loader

For unsupported formats, subclass `LoaderStrategy` and implement `load()`:

```python
from graphrag_sdk import LoaderStrategy
from graphrag_sdk.core.models import DocumentOutput, DocumentInfo

class JsonLinesLoader(LoaderStrategy):
    async def load(self, source: str) -> DocumentOutput:
        import json
        lines: list[str] = []
        with open(source) as f:
            for line in f:
                obj = json.loads(line)
                lines.append(obj.get("text", ""))
        return DocumentOutput(
            text="\n\n".join(lines),
            document_info=DocumentInfo(path=source),
        )


await rag.ingest(source="data.jsonl", loader=JsonLinesLoader())
```

If you can populate `DocumentOutput.elements` with structural elements (headers, tables, paragraphs), structure-aware chunkers like `SentenceTokenCapChunking` will avoid splitting mid-section. For most prose, returning just `text` is fine.

## Skipping the loader entirely

When you already have the text — from a database row, an API response, a stdin pipe — call `ingest(text=...)`:

```python
await rag.ingest(
    text="Alice works at Acme.",
    document_id="row-12345",  # stable id for incremental updates later
)
```

Pass `document_id` if you'll want to `update()` or `delete_document()` later — otherwise the SDK generates a random `text-<8hex>` id.

## See also

- [API Reference → GraphRAG](../api-reference/graphrag#ingest) — `ingest` full signature.
- [API Reference → Ingestion strategies](../api-reference/ingestion-strategies) — every chunker, loader, and extractor.
- [Concepts → Ingestion pipeline](../concepts/ingestion-pipeline) — the four-stage mental model.
