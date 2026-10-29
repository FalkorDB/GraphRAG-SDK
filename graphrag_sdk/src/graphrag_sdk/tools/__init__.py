"""Graph tools for an outside agent (MCP clients, agent frameworks).

The agent plans and answers; the toolkit runs each tool call bounded and
read-only, numbers the evidence it returns, and checks the agent's [n]
citations against it.

Canonical import: ``from graphrag_sdk.tools import GraphRAGToolkit``.
"""

from __future__ import annotations

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.tools.errors import ToolError
from graphrag_sdk.tools.evidence import (
    Evidence,
    EvidenceLedger,
    GroundedAnswer,
    UnsupportedClaim,
    ground_answer,
)
from graphrag_sdk.tools.models import (
    ChunkRef,
    CitationCheckResult,
    ClaimCheck,
    CypherResult,
    CypherRow,
    DocSample,
    DocumentRef,
    EntityCard,
    EntityMatch,
    EntityTypeInfo,
    EvidenceRef,
    Fact,
    FinalizeReport,
    IngestedDocument,
    IngestResult,
    LookupResult,
    OntologyResult,
    PrunedTypes,
    RelationTriple,
    RelationTypeInfo,
    RetrieveResult,
    SampleDocsResult,
    SkillRunResult,
    StatusResult,
    ToolResult,
    TraverseNode,
    TraverseResult,
    WalkPath,
    WalkResult,
)
from graphrag_sdk.tools.specs import TOOL_NAMES, ToolSpec
from graphrag_sdk.tools.toolkit import FinalizePolicy, GraphRAGToolkit

__all__ = [
    "TOOL_NAMES",
    "ChunkRef",
    "CitationCheckResult",
    "ClaimCheck",
    "CypherResult",
    "CypherRow",
    "DocSample",
    "DocumentRef",
    "EntityCard",
    "EntityMatch",
    "EntityTypeInfo",
    "Evidence",
    "EvidenceLedger",
    "EvidenceRef",
    "Fact",
    "FinalizePolicy",
    "FinalizeReport",
    "GraphRAGToolkit",
    "GroundedAnswer",
    "IngestResult",
    "IngestedDocument",
    "LookupResult",
    "OntologyResult",
    "PrunedTypes",
    "ReadOnlyViolation",
    "RelationTriple",
    "RelationTypeInfo",
    "RetrieveResult",
    "SampleDocsResult",
    "SkillRunResult",
    "StatusResult",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "TraverseNode",
    "TraverseResult",
    "UnsupportedClaim",
    "WalkPath",
    "WalkResult",
    "ground_answer",
]
