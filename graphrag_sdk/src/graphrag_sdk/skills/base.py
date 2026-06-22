# GraphRAG SDK — Skills: Base (Phase 3.4)
# A Skill is a composable, high-level reasoning unit built on top of the
# storage + provider primitives. Skills are surfaced three ways: called
# directly, as agentic-retrieval actions (Phase 3.1), and as MCP tools
# (Phase 3.2).

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult

logger = logging.getLogger(__name__)


class Skill(ABC):
    """Abstract base class for high-level graph reasoning skills.

    Args:
        graph_store: Graph data access object (provides ``query_raw``,
            ``weighted_neighbors``, ``get_connected_entities``).
        llm: Optional LLM provider for natural-language synthesis. When
            ``None``, skills return their structured findings without a
            generated summary.
    """

    #: Stable identifier used by registries, the agent, and MCP.
    name: str = "skill"
    #: Human-readable description used in tool schemas.
    description: str = ""

    def __init__(self, graph_store: Any, llm: Any | None = None) -> None:
        self._graph = graph_store
        self._llm = llm

    @abstractmethod
    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        """Execute the skill and return a structured :class:`SkillResult`."""
        ...

    # ── Shared helpers ────────────────────────────────────────────

    async def _rows(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Run a Cypher query and return its raw ``result_set`` rows."""
        try:
            result = await self._graph.query_raw(cypher, params or {})
            return list(getattr(result, "result_set", []) or [])
        except Exception as exc:
            logger.warning("Skill %s query failed: %s", self.name, exc)
            return []

    async def _summarize(self, ctx: Context | None, prompt: str) -> str:
        """Best-effort LLM summary; returns ``""`` when no LLM is configured."""
        if self._llm is None:
            return ""
        try:
            timeout = ctx.provider_timeout_seconds(f"skill {self.name} summary") if ctx else None
            resp = await self._llm.ainvoke(prompt, timeout=timeout)
            return (resp.content or "").strip()
        except Exception as exc:
            logger.warning("Skill %s summary failed: %s", self.name, exc)
            return ""
