# GraphRAG SDK — Ingestion: DOCX Loader
# Pattern: Strategy

from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader


class DocxLoader(DoclingBaseLoader):
    """Load text and structural elements from a DOCX file using Docling."""

    @property
    def extension_name(self) -> str:
        return "docx"
