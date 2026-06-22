# GraphRAG SDK — Skills (Phase 3.4)
# High-level reasoning skills composed from storage + provider primitives.
# Each skill is callable directly, exposed as an agentic-retrieval action,
# and registered as an MCP tool.

from __future__ import annotations

from graphrag_sdk.skills.base import Skill
from graphrag_sdk.skills.contradiction_detection import ContradictionDetectionSkill
from graphrag_sdk.skills.entity_comparison import EntityComparisonSkill
from graphrag_sdk.skills.gap_analysis import GapAnalysisSkill
from graphrag_sdk.skills.impact_analysis import ImpactAnalysisSkill
from graphrag_sdk.skills.timeline_reconstruction import TimelineReconstructionSkill

#: Registry of all built-in skill classes, keyed by stable name.
SKILL_REGISTRY: dict[str, type[Skill]] = {
    EntityComparisonSkill.name: EntityComparisonSkill,
    ImpactAnalysisSkill.name: ImpactAnalysisSkill,
    ContradictionDetectionSkill.name: ContradictionDetectionSkill,
    GapAnalysisSkill.name: GapAnalysisSkill,
    TimelineReconstructionSkill.name: TimelineReconstructionSkill,
}


def build_skill(name: str, graph_store: object, llm: object | None = None) -> Skill:
    """Instantiate a registered skill by name.

    Raises ``KeyError`` with the list of valid names when ``name`` is unknown.
    """
    try:
        skill_cls = SKILL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown skill '{name}'. Available: {sorted(SKILL_REGISTRY)}"
        ) from None
    return skill_cls(graph_store, llm)


__all__ = [
    "Skill",
    "SKILL_REGISTRY",
    "build_skill",
    "EntityComparisonSkill",
    "ImpactAnalysisSkill",
    "ContradictionDetectionSkill",
    "GapAnalysisSkill",
    "TimelineReconstructionSkill",
]
