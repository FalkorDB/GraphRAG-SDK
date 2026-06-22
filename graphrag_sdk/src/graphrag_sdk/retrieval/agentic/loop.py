# GraphRAG SDK — Agentic Retrieval: ReAct loop (Phase 3.1)
# A budget-aware Thought→Action→Observation controller. Reuses existing
# retrieval/storage/skill primitives as tools (see tools.py) and stops on
# a Final Answer, a step cap, or an exhausted latency budget.

from __future__ import annotations

import json
import logging
import re
from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    AgentStep,
    AgentTrace,
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
)
from graphrag_sdk.retrieval.agentic.prompts import render_system_prompt
from graphrag_sdk.retrieval.agentic.tools import ToolRegistry, build_default_registry
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

logger = logging.getLogger(__name__)

_ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_react_step(text: str) -> dict[str, Any]:
    """Parse one ReAct turn into thought/action/action_input/final_answer."""
    final = _FINAL_RE.search(text)
    if final:
        thought = _THOUGHT_RE.search(text)
        return {
            "thought": thought.group(1).strip() if thought else "",
            "final_answer": final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    input_m = _ACTION_INPUT_RE.search(text)

    action = ""
    if action_m:
        # Take only the first line of the action and strip stray input text.
        action = action_m.group(1).splitlines()[0].strip()
        action = re.split(r"\bAction\s*Input\b", action, flags=re.IGNORECASE)[0].strip()

    action_input: dict[str, Any] = {}
    if input_m:
        try:
            parsed = json.loads(input_m.group(1).strip())
            if isinstance(parsed, dict):
                action_input = parsed
        except (ValueError, TypeError):
            action_input = {}

    return {
        "thought": thought_m.group(1).splitlines()[0].strip() if thought_m else "",
        "action": action,
        "action_input": action_input,
    }


class AgenticRetrieval(RetrievalStrategy):
    """ReAct-style agentic retrieval strategy.

    Drives an LLM tool loop that searches, traverses, runs Cypher, and
    invokes skills until it produces a Final Answer or hits the step /
    latency budget. The collected observations become retrieval items and
    the full reasoning trace is exposed in ``RetrieverResult.metadata``.

    Args:
        llm: LLM provider (uses ``ainvoke``).
        registry: Tool registry. If omitted, a default registry is built
            from ``strategy`` + ``graph_store``.
        strategy: Inner retrieval strategy backing the ``search`` tool.
        graph_store: GraphStore backing ``cypher``/``traverse``/skills.
        max_steps: Hard cap on Thought/Action iterations.
    """

    def __init__(
        self,
        llm: Any,
        *,
        registry: ToolRegistry | None = None,
        strategy: Any | None = None,
        graph_store: Any | None = None,
        vector_store: Any | None = None,
        max_steps: int = 6,
    ) -> None:
        super().__init__(graph_store=graph_store, vector_store=vector_store)
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._llm = llm
        self._max_steps = max_steps
        self._registry = registry or build_default_registry(
            strategy=strategy,
            graph_store=graph_store,
            llm=llm,
        )

    async def _execute(self, query: str, ctx: Context, **kwargs: Any) -> RawSearchResult:
        system = render_system_prompt(
            tool_descriptions=self._registry.describe(),
            tool_names=", ".join(self._registry.names()),
        )
        scratchpad = f"Question: {query}\n"
        trace = AgentTrace()
        observations: list[str] = []
        stop_reason = "max_steps"

        for step_idx in range(self._max_steps):
            if ctx.budget_exceeded:
                stop_reason = "budget_exceeded"
                ctx.log("Agentic loop stopped: latency budget exhausted")
                break

            prompt = f"{system}\n\n{scratchpad}\nThought:"
            timeout = ctx.provider_timeout_seconds(f"agentic step {step_idx}")
            response = await self._llm.ainvoke(prompt, timeout=timeout)
            text = response.content or ""
            parsed = parse_react_step(text)

            if "final_answer" in parsed:
                trace.steps.append(AgentStep(index=step_idx, thought=parsed.get("thought", "")))
                trace.answer = parsed["final_answer"]
                stop_reason = "final_answer"
                ctx.log(f"Agentic loop produced final answer at step {step_idx}")
                break

            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})
            if not action:
                stop_reason = "no_action"
                ctx.log("Agentic loop stopped: model emitted no action")
                break

            observation = await self._registry.run(action, action_input, ctx)
            observations.append(observation)
            trace.steps.append(
                AgentStep(
                    index=step_idx,
                    thought=parsed.get("thought", ""),
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            )
            scratchpad += (
                f"Thought: {parsed.get('thought', '')}\n"
                f"Action: {action}\n"
                f"Action Input: {json.dumps(action_input)}\n"
                f"Observation: {observation}\n"
            )

        trace.stop_reason = stop_reason
        records: list[Any] = list(observations)
        if trace.answer:
            records.append(f"Answer: {trace.answer}")
        return RawSearchResult(
            records=records,
            metadata={
                "agent_trace": trace.model_dump(),
                "stop_reason": stop_reason,
                "num_steps": trace.num_steps,
                "answer": trace.answer,
            },
        )

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        items = [
            RetrieverResultItem(content=str(rec), metadata={"source": "agentic"})
            for rec in raw.records
        ]
        return RetrieverResult(items=items, metadata=raw.metadata)
