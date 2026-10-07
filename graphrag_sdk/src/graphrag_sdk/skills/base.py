# GraphRAG SDK — Skills: Base (Phase 3.4)
# A Skill is a composable, high-level reasoning unit built on top of the
# storage + provider primitives. Skills are surfaced three ways: called
# directly (GraphRAG.run_skill), as agentic-retrieval tools, and as MCP tools.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import SkillResult

logger = logging.getLogger(__name__)


def skill_parameters(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """JSON Schema object for a skill's parameters (no extra keys allowed)."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


#: Schema snippet for a parameter naming one entity.
ENTITY_PARAM = {"type": "string", "description": "Entity name or id."}


class Skill(ABC):
    """Abstract base class for high-level graph reasoning skills.

    Subclasses set ``name``, ``description`` and ``parameters`` (a JSON
    Schema object) and implement :meth:`run`. ``parameters`` is what the
    agent and MCP clients see, and :meth:`GraphRAG.run_skill` checks
    arguments against it.

    Args:
        graph_store: Graph data access object (provides ``query_raw``,
            ``weighted_neighbors``, ``pagerank``).
        llm: Optional LLM provider for natural-language synthesis. When
            ``None``, skills return their structured findings without a
            generated summary.
    """

    #: Stable identifier used by registries, the agent, and MCP.
    name: str = "skill"
    #: Human-readable description used in tool schemas.
    description: str = ""
    #: JSON Schema object describing the skill's parameters.
    parameters: dict[str, Any] = skill_parameters({})

    def __init__(self, graph_store: Any, llm: Any | None = None) -> None:
        self._graph = graph_store
        self._llm = llm

    @abstractmethod
    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        """Execute the skill and return a structured :class:`SkillResult`."""
        raise NotImplementedError("Subclasses must implement 'run'.")

    # ── Shared helpers ────────────────────────────────────────────

    @staticmethod
    def _int_param(params: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
        """Read an int parameter, clamped to ``[low, high]``.

        Raises ``ValueError`` when the value is not a number, so a typo is
        reported rather than silently replaced by the default.
        """
        value = params.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"'{key}' must be an integer")
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be an integer, got {value!r}") from None
        return max(low, min(high, number))

    async def _rows(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        """Run a read-only Cypher query and return its raw ``result_set`` rows."""
        try:
            try:
                result = await self._graph.query_raw(cypher, params or {}, read_only=True)
            except TypeError:  # graph stores without the read_only keyword
                result = await self._graph.query_raw(cypher, params or {})
            return list(getattr(result, "result_set", []) or [])
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.warning("Skill %s query failed: %s", self.name, exc)
            return []

    async def _resolve_entity(self, ref: Any, param: str = "entity") -> str:
        """Return the id of the one entity ``ref`` names (its id or its name).

        Exact id or name first, then a case-insensitive name match. Raises
        ``ValueError`` when nothing or more than one entity matches, naming
        the candidates so the caller can retry with an id.
        """
        text = str(ref or "").strip()
        if not text:
            raise ValueError(f"{self.name} requires '{param}'")
        for cypher in (
            "MATCH (e:__Entity__) WHERE e.id = $ref OR e.name = $ref RETURN e.id, e.name LIMIT 6",
            "MATCH (e:__Entity__) WHERE toLower(toString(e.name)) = toLower($ref) "
            "RETURN e.id, e.name LIMIT 6",
        ):
            rows = [r for r in await self._rows(cypher, {"ref": text}) if r and r[0] is not None]
            ids = list(dict.fromkeys(str(r[0]) for r in rows))
            if len(ids) == 1:
                return ids[0]
            if len(ids) > 1:
                listed = "; ".join(f'"{r[1]}" (id {r[0]})' for r in rows[:5])
                raise ValueError(f"'{param}' {text!r} matches several entities: {listed}")
        raise ValueError(f"'{param}': no entity is called {text!r}")

    async def _summarize(self, ctx: Context | None, prompt: str) -> str:
        """Best-effort LLM summary; returns ``""`` when no LLM is configured."""
        if self._llm is None:
            return ""
        try:
            timeout = ctx.provider_timeout_seconds(f"skill {self.name} summary") if ctx else None
            resp = await self._llm.ainvoke(prompt, timeout=timeout)
            return (resp.content or "").strip()
        except LatencyBudgetExceededError:
            raise
        except Exception as exc:
            logger.warning("Skill %s summary failed: %s", self.name, exc)
            return ""
