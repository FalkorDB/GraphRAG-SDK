# GraphRAG SDK — Skills (Phase 3.4)
# High-level reasoning skills composed from storage + provider primitives.
# Each skill is callable directly, exposed as an agentic-retrieval tool,
# and registered as an MCP tool.

from __future__ import annotations

import re

from graphrag_sdk.skills.base import ENTITY_PARAM, Skill, skill_parameters
from graphrag_sdk.skills.contradiction_detection import ContradictionDetectionSkill
from graphrag_sdk.skills.entity_comparison import EntityComparisonSkill
from graphrag_sdk.skills.gap_analysis import GapAnalysisSkill
from graphrag_sdk.skills.impact_analysis import ImpactAnalysisSkill
from graphrag_sdk.skills.timeline_reconstruction import TimelineReconstructionSkill

_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
#: Names of the agent's built-in tools; a skill registered under one would
#: collide with it in every agent's tool registry.
_RESERVED_NAMES = frozenset({"search", "query_graph", "traverse", "lookup_entity"})

#: Registry of skill classes, keyed by stable name. Add your own with
#: :func:`register_skill`; the agent, MCP and ``GraphRAG.run_skill`` all
#: read from it.
SKILL_REGISTRY: dict[str, type[Skill]] = {
    EntityComparisonSkill.name: EntityComparisonSkill,
    ImpactAnalysisSkill.name: ImpactAnalysisSkill,
    ContradictionDetectionSkill.name: ContradictionDetectionSkill,
    GapAnalysisSkill.name: GapAnalysisSkill,
    TimelineReconstructionSkill.name: TimelineReconstructionSkill,
}


def register_skill(skill_cls: type[Skill], *, replace: bool = False) -> type[Skill]:
    """Add a skill class to :data:`SKILL_REGISTRY` (usable as a decorator).

    The class must subclass :class:`Skill`, have a tool-safe ``name``
    (letters, digits, ``_`` and ``-``) and a JSON Schema object as
    ``parameters``. Registering a taken name raises unless ``replace=True``.

    Example::

        @register_skill
        class OwnersSkill(Skill):
            name = "owners"
            description = "List the owners of an entity."
            parameters = skill_parameters({"entity": ENTITY_PARAM}, ["entity"])

            async def run(self, ctx=None, **params): ...
    """
    if not (isinstance(skill_cls, type) and issubclass(skill_cls, Skill)):
        raise TypeError("register_skill expects a Skill subclass")
    name = getattr(skill_cls, "name", "")
    if not isinstance(name, str) or not _SKILL_NAME_RE.match(name) or name == "skill":
        raise ValueError(
            f"Invalid skill name {name!r}: set a unique `name` of letters, digits, _ or -"
        )
    params = getattr(skill_cls, "parameters", None)
    if not isinstance(params, dict) or params.get("type") != "object":
        raise ValueError(f"Skill {name!r}: `parameters` must be a JSON Schema object")
    if name in _RESERVED_NAMES:
        raise ValueError(f"Skill name {name!r} is reserved for a built-in agent tool")
    if name in SKILL_REGISTRY and SKILL_REGISTRY[name] is not skill_cls and not replace:
        raise ValueError(f"A skill named {name!r} is already registered (pass replace=True)")
    SKILL_REGISTRY[name] = skill_cls
    return skill_cls


def build_skill(name: str, graph_store: object, llm: object | None = None) -> Skill:
    """Instantiate a registered skill by name.

    Raises ``KeyError`` with the list of valid names when ``name`` is unknown.
    """
    try:
        skill_cls = SKILL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown skill '{name}'. Available: {sorted(SKILL_REGISTRY)}") from None
    return skill_cls(graph_store, llm)


__all__ = [
    "ENTITY_PARAM",
    "Skill",
    "SKILL_REGISTRY",
    "build_skill",
    "register_skill",
    "skill_parameters",
    "EntityComparisonSkill",
    "ImpactAnalysisSkill",
    "ContradictionDetectionSkill",
    "GapAnalysisSkill",
    "TimelineReconstructionSkill",
]
