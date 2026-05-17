# GraphRAG SDK — Ingestion: PPTX Loader
# Pattern: Strategy

from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader


class PptxLoader(DoclingBaseLoader):
    """Load text and structural elements from a PPTX file using Docling."""

    @property
    def extension_name(self) -> str:
        return "pptx"
