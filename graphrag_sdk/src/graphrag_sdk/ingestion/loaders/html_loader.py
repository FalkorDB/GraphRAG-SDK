# GraphRAG SDK — Ingestion: HTML Loader
# Pattern: Strategy

from graphrag_sdk.ingestion.loaders.docling_base import DoclingBaseLoader


class HtmlLoader(DoclingBaseLoader):
    """Load text and structural elements from an HTML/XHTML file using Docling."""

    @property
    def extension_name(self) -> str:
        return "html"
