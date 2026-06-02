# GraphRAG SDK — Discovery package
#
# Auto-discover an ontology from a corpus. Public entry points:
#
#   - ``Ontology.from_sources`` (in graphrag_sdk.core.models) →
#     ``discover_ontology`` here. Bootstrap or offline refresh.
#   - ``GraphRAG.suggest_schema_extensions`` (in graphrag_sdk.api.main) →
#     ``suggest_extensions`` here. Live-graph delta proposals.

from __future__ import annotations

from graphrag_sdk.discovery.instructor import extract_with_retry
from graphrag_sdk.discovery.pipeline import (
    discover_ontology,
    suggest_extensions,
)
from graphrag_sdk.discovery.proposal import (
    ChunkProposal,
    DocSummary,
    NormalizedDraft,
    OntologyDiscoveryError,
    SchemaExtensionProposal,
)

__all__ = [
    "ChunkProposal",
    "DocSummary",
    "NormalizedDraft",
    "OntologyDiscoveryError",
    "SchemaExtensionProposal",
    "discover_ontology",
    "extract_with_retry",
    "suggest_extensions",
]
