# GraphRAG SDK — Connectors
#
# Sync databases, file stores and apps into the graph through existing
# frameworks: dlt for the sources it ships, LlamaIndex readers for the rest.
# The dlt helpers are imported on first use, so the plain-iterable functions
# work without the extra installed.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphrag_sdk.connectors.sync import (
    SourceDocument,
    SyncReport,
    SyncResult,
    document_id_for,
    sync,
    sync_documents,
    sync_table,
)

if TYPE_CHECKING:
    from graphrag_sdk.connectors.sources import file_documents, llamaindex_documents

__all__ = [
    "SourceDocument",
    "SyncReport",
    "SyncResult",
    "document_id_for",
    "file_documents",
    "llamaindex_documents",
    "sync",
    "sync_documents",
    "sync_table",
]


def __getattr__(name: str) -> Any:
    if name in ("file_documents", "llamaindex_documents"):
        from graphrag_sdk.connectors import sources

        return getattr(sources, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
