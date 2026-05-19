# GraphRAG SDK — Ingestion: XLSX Loader
# Pattern: Strategy

from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader


class XlsxLoader(DoclingBaseLoader):
    """Load text and structural elements from an XLSX file using Docling."""

    @property
    def extension_name(self) -> str:
        return "xlsx"
