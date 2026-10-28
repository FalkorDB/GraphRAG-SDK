# GraphRAG SDK — Connectors: dlt resources that yield documents
#
# dlt ships sources for databases (sql_database), file stores (filesystem: S3,
# GCS, Azure, SFTP, Google Drive) and REST APIs (rest_api). Rows from those need
# nothing here: sync() writes them as tables. What dlt lacks is document content
# from apps such as Notion, Confluence and SharePoint, and text out of files, so
# these two helpers turn a LlamaIndex reader or a dlt filesystem listing into
# SourceDocument items.

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

from graphrag_sdk.connectors.sync import SourceDocument

try:
    import dlt
    from dlt.extract import DltResource
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "graphrag_sdk.connectors.sources needs dlt. Install it with: pip install graphrag-sdk[dlt]"
    ) from exc

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = (".md", ".markdown", ".txt", ".text", ".rst")


def llamaindex_documents(
    reader: Any,
    *,
    name: str,
    id_from: Callable[[Any], str],
    **load_kwargs: Any,
) -> DltResource:
    """A dlt resource of the documents a LlamaIndex reader loads.

    LlamaIndex readers cover apps dlt has no document source for, such as
    Notion pages (``llama-index-readers-notion``), Confluence and SharePoint.
    Install the reader's own package; this SDK does not depend on LlamaIndex.

    Example::

        from llama_index.readers.notion import NotionPageReader

        notion = llamaindex_documents(
            NotionPageReader(integration_token=token),
            name="notion", id_from=lambda doc: doc.id_, page_ids=ids
        )
        await sync(rag, notion)

    Args:
        reader: Any object with LlamaIndex's ``load_data(**kwargs)`` method.
        name: The source name. Each document's graph id is ``<name>:<id>``.
        id_from: Required callback returning a stable source id. Use ``doc.id_``
            only when the reader sets it to the source's id, as Notion does.
            Other readers may generate random ids; use their metadata instead.
        **load_kwargs: Passed to ``reader.load_data``.
    """

    @dlt.resource(name=name)
    def documents() -> Iterator[SourceDocument]:
        for doc in reader.load_data(**load_kwargs):
            doc_id = id_from(doc)
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("id_from must return a non-empty stable source id")
            yield SourceDocument(id=str(doc_id), text=doc.text, metadata=dict(doc.metadata or {}))

    return documents()


def file_documents(files: DltResource, *, name: str) -> DltResource:
    """Documents read from the files a dlt ``filesystem`` resource lists.

    Text files (``.md``, ``.txt``, ``.rst``) are decoded as UTF-8 and PDFs are
    read with the SDK's PDF loader (``graphrag-sdk[pdf]``). Other files are
    skipped and logged.

    Example::

        from dlt.sources.filesystem import filesystem

        drive = file_documents(
            filesystem(bucket_url="gdrive://<folder id>", file_glob="**/*.md"),
            name="drive",
        )
        await sync(rag, drive)

    Args:
        files: A dlt ``filesystem`` resource. Its credentials are dlt's
            (arguments, environment or ``.dlt/secrets.toml``).
        name: The source name. Each document's graph id is ``<name>:<file URL>``.
    """

    @dlt.transformer(name=name)
    def documents(items: Any) -> Iterator[SourceDocument]:
        # dlt's filesystem resource yields pages of files; a custom listing may
        # yield them one at a time.
        for item in [items] if isinstance(items, dict) else items:
            text = _file_text(item)
            if text is None:
                continue
            modified = item.get("modification_date")
            yield SourceDocument(
                id=item["file_url"],
                text=text,
                metadata={
                    "source_url": item["file_url"],
                    "file_name": item.get("file_name"),
                    "mime_type": item.get("mime_type"),
                    "modified": modified,
                },
            )

    return files | documents


def _file_text(item: Any) -> str | None:
    suffix = os.path.splitext(item["file_name"])[1].lower()
    if suffix in _TEXT_SUFFIXES:
        data: bytes = item.read_bytes()
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader

        # Fetched before the file exists, so a network error leaves nothing
        # behind. Closed before reading: Windows refuses to reopen a file that
        # is still open, which NamedTemporaryFile's delete-on-close would need.
        content: bytes = item.read_bytes()
        fd, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            return PdfLoader()._load_sync(path).text
        finally:
            os.unlink(path)
    logger.info("skipped %s: no text reader for %s files", item["file_url"], suffix or "these")
    return None
