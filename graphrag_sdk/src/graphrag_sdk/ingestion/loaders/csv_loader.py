# GraphRAG SDK — Ingestion: CSV Loader
# Pattern: Strategy

from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader


class CsvLoader(DoclingBaseLoader):
    """Load text and structural elements from a CSV file using Docling."""

    @property
    def extension_name(self) -> str:
        return "csv"
