# GraphRAG SDK — Agentic Retrieval: run limits
# One place for every bound on an agent run: how many model turns and tool
# calls it may make, how much text goes back to the model, and the ceilings
# on what each graph tool may do. Model-supplied tool arguments are clamped
# to these ceilings, never trusted.

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class AgentLimits:
    """Bounds on one agentic retrieval run.

    Run:
        max_steps: Model turns that may call tools. When reached, the model
            is asked once more to answer without tools.
        max_tool_calls: Tool calls in the whole run. Further calls are refused
            and the model is asked to answer.
        max_calls_per_turn: Tool calls run from one model response; the rest
            of that response's calls are refused.
        max_observation_chars: Characters of one tool result sent back to the
            model (the evidence keeps the full text).
        max_history_turns: Earlier user/assistant exchanges kept.
        max_trace_chars: Characters kept per trace step (observation and
            JSON-encoded arguments).

    Tools (ceilings; a model asking for more gets the ceiling):
        search_max_items: Numbered results one search returns.
        search_max_limit: Ceiling for search's retrieval-limit arguments
            (``chunk_top_k``, ``max_entities`` ...).
        query_max_rows / query_max_cell_chars / query_timeout_ms /
        query_attempts: ``query_graph`` row cap, cell truncation, database
            timeout and generation attempts.
        traverse_max_depth / traverse_max_expansions /
        traverse_max_timeout_ms: ``traverse`` ceilings.
    """

    max_steps: int = 6
    max_tool_calls: int = 12
    max_calls_per_turn: int = 4
    max_observation_chars: int = 8000
    max_history_turns: int = 5
    max_trace_chars: int = 2000

    search_max_items: int = 40
    search_max_limit: int = 50

    query_max_rows: int = 25
    query_max_cell_chars: int = 200
    query_timeout_ms: int = 5000
    query_attempts: int = 3

    traverse_max_depth: int = 5
    traverse_max_expansions: int = 200
    traverse_max_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            minimum = 0 if f.name == "max_history_turns" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"AgentLimits.{f.name} must be an int >= {minimum}, got {value!r}")
        if self.traverse_max_timeout_ms < 50:
            raise ValueError("AgentLimits.traverse_max_timeout_ms must be >= 50")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def truncate(text: str, max_chars: int) -> str:
    """``text`` cut to ``max_chars`` with a note saying how much was dropped."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…[truncated {len(text) - max_chars} characters]"
