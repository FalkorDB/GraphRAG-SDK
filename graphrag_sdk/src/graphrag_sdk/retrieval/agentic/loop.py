# GraphRAG SDK — Agentic Retrieval: agent loop (Phase 3.1)
# A budget-aware tool loop. With an LLM that supports native tool calling
# the model requests tools as structured calls; otherwise the loop falls back
# to text ReAct (Thought / Action / Observation) and parses the reply. Both
# paths share the tool registry, the trace and the stop rules.

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, Literal

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError
from graphrag_sdk.core.models import (
    AgentStep,
    AgentTrace,
    ChatMessage,
    Ontology,
    RawSearchResult,
    RetrieverResult,
    RetrieverResultItem,
    ToolCall,
)
from graphrag_sdk.retrieval.agentic.citations import (
    DEFAULT_UNGROUNDED_ANSWER,
    GroundedAnswer,
    GroundingPolicy,
    ground_answer,
)
from graphrag_sdk.retrieval.agentic.graph_tools import build_default_registry
from graphrag_sdk.retrieval.agentic.limits import AgentLimits, truncate
from graphrag_sdk.retrieval.agentic.prompts import (
    CITATION_REPAIR_PROMPT,
    FINAL_ANSWER_NUDGE,
    REACT_FORMAT_REMINDER,
    render_native_system_prompt,
    render_system_prompt,
)
from graphrag_sdk.retrieval.agentic.tools import (
    Evidence,
    ToolContext,
    ToolRegistry,
    ToolResult,
    refused,
)
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy

logger = logging.getLogger(__name__)

AgentMode = Literal["auto", "native", "react"]

_ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(\{)", re.IGNORECASE)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

#: Grounding verdicts a citation repair can fix: the tools returned evidence
#: but the answer cites none of it (or cites numbers that do not exist, or is
#: empty because the run ended before the model answered).
_REPAIRABLE = frozenset({"no_citations", "invalid_citations", "empty"})
#: Upper bound on the evidence text sent with a citation repair, and on one
#: evidence item within it.
REPAIR_MAX_EVIDENCE_CHARS = 12_000
REPAIR_MAX_ITEM_CHARS = 1_500


def _normalize_query(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def render_repair_evidence(
    items: Sequence[Evidence],
    *,
    first: Sequence[int] = (),
    max_chars: int = REPAIR_MAX_EVIDENCE_CHARS,
    max_item_chars: int = REPAIR_MAX_ITEM_CHARS,
) -> str:
    """The numbered evidence for a citation repair, within ``max_chars``.

    Items whose numbers are in ``first`` lead (the passages a repair search
    just found for the question); the rest follow in the order the tools
    returned them. Items that no longer fit are left out whole.
    """
    lead = [ev for n in first for ev in items if ev.n == n]
    lead_ns = {ev.n for ev in lead}
    ordered = lead + [ev for ev in items if ev.n not in lead_ns]
    lines: list[str] = []
    used = 0
    for ev in ordered:
        prefix = f"(source: {ev.source}) " if ev.source else ""
        line = f"[{ev.n}] {prefix}{truncate(ev.content, max_item_chars)}"
        if lines and used + len(line) > max_chars:
            continue
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _parse_action_input(text: str, start: int) -> dict[str, Any]:
    """Decode the JSON object starting at ``start`` (an opening brace).

    Uses ``raw_decode`` so parsing stops at the object's matching close brace —
    trailing prose or a hallucinated ``Observation:`` line can't corrupt the
    arguments the way a greedy first-``{``-to-last-``}`` capture did.
    """
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, start)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
        action_input = _parse_action_input(text, input_m.start(1))

    return {
        "thought": thought_m.group(1).splitlines()[0].strip() if thought_m else "",
        "action": action,
        "action_input": action_input,
    }


def _echoable(call: ToolCall) -> ToolCall:
    """The call as it is sent back to the provider in the conversation.

    Arguments that were not valid JSON are echoed as ``{}``: some provider
    adapters re-parse earlier calls' arguments and would fail the next turn.
    The refusal sent as this call's result still tells the model what was
    wrong.
    """
    if call.parse_error:
        return call.model_copy(update={"raw_arguments": "{}", "arguments": {}})
    return call


def history_as_messages(
    history: Sequence[ChatMessage | dict[str, Any]] | None,
    *,
    max_turns: int = 5,
) -> list[ChatMessage]:
    """Earlier conversation as text-only user/assistant messages.

    Keeps the last ``max_turns`` exchanges. System messages, tool turns and
    empty messages are dropped: earlier turns' tool results are large and
    are not re-sent, and the agent's own system prompt always applies.
    """
    if not history or max_turns < 1:
        return []
    kept: list[ChatMessage] = []
    role: Any
    content: Any
    tool_calls: Any
    for msg in history:
        if isinstance(msg, ChatMessage):
            role, content, tool_calls = msg.role, msg.content, msg.tool_calls
        elif isinstance(msg, dict):
            role, content, tool_calls = msg.get("role"), msg.get("content"), None
        else:
            continue
        if role not in ("user", "assistant") or tool_calls:
            continue
        text = str(content or "").strip()
        if text:
            kept.append(ChatMessage(role=role, content=text))
    return kept[-(max_turns * 2) :]


class AgenticRetrieval(RetrievalStrategy):
    """Agentic retrieval: an LLM tool loop over the knowledge graph.

    The model searches, queries and walks the graph through tools until it
    answers or reaches the step / latency budget. The collected tool results
    become retrieval items and the full trace is exposed in
    ``RetrieverResult.metadata``.

    Two loop modes share the same tools and stop rules:

    - ``native``: structured tool calls via ``llm.ainvoke_with_tools``.
    - ``react``: text Thought / Action / Observation turns parsed from the
      reply, for providers without native tool calling.

    ``mode="auto"`` (the default) picks ``native`` when
    ``llm.supports_tool_calling`` is true, otherwise ``react``.

    Args:
        llm: LLM provider.
        registry: Tool registry. If omitted, a default registry is built
            from ``strategy`` + ``graph_store``.
        strategy: Inner retrieval strategy backing the ``search`` tool.
        graph_store: GraphStore backing the graph tools and skills.
        vector_store: Optional vector store (kept for API symmetry).
        max_steps: Shortcut for ``limits.max_steps``: model turns that may
            call tools. When reached the model is asked once more to answer,
            without tools.
        mode: ``"auto"``, ``"native"`` or ``"react"``.
        system_prompt: Replace the built-in system prompt (native mode) or
            prepend to it (react mode, which needs the format instructions).
        history_turns: Shortcut for ``limits.max_history_turns``.
        ontology: Ontology the ``query_graph`` tool writes Cypher against.
            The facade keeps it current through :meth:`set_ontology`.
        grounding: How the answer's ``[N]`` citations are enforced:
            ``"strict"`` (default) replaces an answer that cites no real tool
            result with ``ungrounded_answer``; ``"annotate"`` keeps it and
            reports ``grounded=False``; ``"off"`` skips the check.
        ungrounded_answer: The answer shown instead of an ungrounded one.
        repair_citations: When the answer cites none of the evidence the
            tools returned, ask the model once to rewrite it from that
            evidence with citations before the grounding check decides
            (default ``True``; never applies with ``grounding="off"``). If
            the question as asked was never searched, it is searched first so
            the rewrite has its passages to cite. The outcome is in
            ``metadata["citation_repair"]``.
        limits: Every bound on the run and the default tools (see
            :class:`AgentLimits`): turns, tool calls (in total and per
            turn), text returned to the model, trace size, and the
            ceilings of search / query_graph / traverse.

    The run's answer is final: ``metadata["answer"]`` is what the agent
    concluded (after the grounding check), ``metadata["citations"]`` the
    evidence it cites, and :meth:`GraphRAG.completion` returns it as-is
    instead of generating a second answer. The result items are the numbered
    evidence (``"[N] ..."``) the tools returned.

    Pass ``history=[...]`` to :meth:`search` to give the agent the
    conversation so far (``ChatMessage`` objects or ``{"role", "content"}``
    dicts).
    """

    #: The facade forwards conversation history to strategies that set this.
    accepts_history = True

    def __init__(
        self,
        llm: Any,
        *,
        registry: ToolRegistry | None = None,
        strategy: Any | None = None,
        graph_store: Any | None = None,
        vector_store: Any | None = None,
        max_steps: int | None = None,
        mode: AgentMode = "auto",
        system_prompt: str | None = None,
        history_turns: int | None = None,
        ontology: Ontology | None = None,
        grounding: GroundingPolicy = "strict",
        ungrounded_answer: str = DEFAULT_UNGROUNDED_ANSWER,
        repair_citations: bool = True,
        limits: AgentLimits | None = None,
    ) -> None:
        super().__init__(graph_store=graph_store, vector_store=vector_store)
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if mode not in ("auto", "native", "react"):
            raise ValueError("mode must be 'auto', 'native' or 'react'")
        limits = limits or AgentLimits()
        overrides: dict[str, int] = {}
        if max_steps is not None:
            overrides["max_steps"] = max_steps
        if history_turns is not None:
            overrides["max_history_turns"] = history_turns
        if overrides:
            limits = AgentLimits(**{**limits.to_dict(), **overrides})
        self._limits = limits
        self._llm = llm
        self._max_steps = limits.max_steps
        self._mode = mode
        self._system_prompt = system_prompt
        self._history_turns = limits.max_history_turns
        if grounding not in ("strict", "annotate", "off"):
            raise ValueError("grounding must be 'strict', 'annotate' or 'off'")
        self._grounding: GroundingPolicy = grounding
        self._ungrounded_answer = ungrounded_answer
        self._repair_citations = repair_citations
        self._ontology = ontology
        self._strategy = strategy
        self._registry = (
            registry
            if registry is not None
            else build_default_registry(
                strategy=strategy,
                graph_store=graph_store,
                llm=llm,
                ontology_getter=lambda: self._ontology,
                limits=limits,
            )
        )

    def set_ontology(self, ontology: Any) -> None:
        """Adopt the current ontology (for ``query_graph``) and pass it on."""
        self._ontology = ontology
        inner = getattr(self._strategy, "set_ontology", None)
        if callable(inner):
            inner(ontology)

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def limits(self) -> AgentLimits:
        return self._limits

    def resolved_mode(self) -> Literal["native", "react"]:
        """The loop mode a run will use with the configured LLM."""
        if self._mode == "native":
            return "native"
        if self._mode == "react":
            return "react"
        return "native" if getattr(self._llm, "supports_tool_calling", False) else "react"

    async def _execute(self, query: str, ctx: Context, **kwargs: Any) -> RawSearchResult:
        if len(self._registry) == 0:
            raise ValueError("AgenticRetrieval needs at least one tool")
        history = history_as_messages(kwargs.get("history"), max_turns=self._history_turns)
        tctx = ToolContext(ctx=ctx)
        mode = self.resolved_mode()
        trace = AgentTrace(mode=mode)
        observations: list[str] = []
        if mode == "native":
            await self._run_native(query, history, ctx, tctx, trace, observations)
        else:
            await self._run_react(query, history, ctx, tctx, trace, observations)

        grounded = self._ground(trace.answer, tctx, trace)
        repair: dict[str, Any] = {"attempted": False}
        if self._should_repair(grounded, ctx, trace):
            grounded, repair = await self._repair(query, grounded, ctx, tctx, trace, observations)
        if not grounded.grounded:
            ctx.log(f"Agentic answer not grounded ({grounded.reason})")
        records = [ev.to_dict() for ev in tctx.evidence.items]
        return RawSearchResult(
            records=records,
            metadata={
                "answer": grounded.answer,
                "answer_is_final": True,
                "raw_answer": grounded.raw_answer,
                "grounded": grounded.grounded,
                "grounding_reason": grounded.reason,
                "citations": [ev.to_dict() for ev in grounded.citations(tctx.evidence)],
                "invalid_citations": grounded.invalid,
                "citation_repair": repair,
                "agent_trace": trace.model_dump(),
                "agent_mode": mode,
                "stop_reason": trace.stop_reason,
                "num_steps": trace.num_steps,
                "evidence": records,
                "generated_cypher": list(tctx.state.get("generated_cypher", [])),
                "tool_calls": tctx.state.get("tool_calls", 0),
                "limits": self._limits.to_dict(),
            },
        )

    # ── Grounding and citation repair ─────────────────────────────

    def _ground(self, answer: str, tctx: ToolContext, trace: AgentTrace) -> GroundedAnswer:
        return ground_answer(
            answer,
            tctx.evidence,
            tools_called=bool(trace.steps),
            policy=self._grounding,
            ungrounded_answer=self._ungrounded_answer,
        )

    def _should_repair(self, grounded: GroundedAnswer, ctx: Context, trace: AgentTrace) -> bool:
        return (
            self._repair_citations
            and self._grounding != "off"
            and not grounded.grounded
            and grounded.reason in _REPAIRABLE
            and not ctx.budget_exceeded
            and trace.stop_reason != "budget_exceeded"
        )

    async def _repair(
        self,
        query: str,
        grounded: GroundedAnswer,
        ctx: Context,
        tctx: ToolContext,
        trace: AgentTrace,
        observations: list[str],
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        """One rewrite of an uncited answer from the run's numbered evidence.

        When the question itself was never searched (the model went straight
        to a graph query or a walk, or searched only reworded queries), it is
        searched first: the question as asked is what a normal completion
        retrieves with, and its passages are often the ones the answer needs.
        The rewrite goes through the same grounding check as any answer, so a
        reply that still cites nothing stays ungrounded.
        """
        info: dict[str, Any] = {
            "attempted": True,
            "searched": False,
            "grounded": False,
            "reason_before": grounded.reason,
        }
        try:
            first: list[int] = []
            asked = _normalize_query(query)
            searched = any(
                step.action == "search"
                and step.status == "ok"
                and _normalize_query(str(step.action_input.get("query", ""))) == asked
                for step in trace.steps
            )
            if not searched and "search" in self._registry.names():
                result = await self._call_tool("search", {"query": query}, tctx)
                self._record(
                    trace,
                    observations,
                    index=len(trace.steps),
                    thought="citation repair: search the question for passages to cite",
                    action="search",
                    action_input={"query": query},
                    result=result,
                )
                if result.status == "ok":
                    info["searched"] = True
                    first = list((result.data or {}).get("evidence", []))
            if not len(tctx.evidence):
                info["skipped"] = "no evidence to cite"
                return grounded, info
            prompt = CITATION_REPAIR_PROMPT.format(
                question=query,
                evidence=render_repair_evidence(tctx.evidence.items, first=first),
                draft=grounded.raw_answer or "(no answer)",
            )
            response = await self._llm.ainvoke(
                prompt, timeout=ctx.provider_timeout_seconds("agentic citation repair")
            )
        except LatencyBudgetExceededError:
            info["skipped"] = "latency budget exhausted"
            return grounded, info
        except Exception as exc:  # noqa: BLE001 - a failed repair keeps the verdict
            logger.warning("Agentic citation repair failed: %s", exc)
            info["error"] = f"{type(exc).__name__}: {exc}"[:300]
            return grounded, info

        repaired = self._ground((response.content or "").strip(), tctx, trace)
        info["grounded"] = repaired.grounded
        if repaired.grounded:
            ctx.log("Agentic answer grounded after citation repair")
            # ``raw_answer`` stays the model's first answer, for inspection;
            # the answer shown is the cited rewrite.
            repaired.raw_answer = grounded.raw_answer
            return repaired, info
        return grounded, info

    # ── Shared helpers ────────────────────────────────────────────

    async def _call_tool(
        self,
        name: str,
        args: dict[str, Any],
        tctx: ToolContext,
    ) -> ToolResult:
        tctx.ctx.ensure_budget(f"agent tool {name}")
        used = tctx.state.get("tool_calls", 0)
        if used >= self._limits.max_tool_calls:
            return refused(
                name,
                f"the limit of {self._limits.max_tool_calls} tool calls for this question is "
                "reached; answer from the results you already have",
            )
        tctx.state["tool_calls"] = used + 1
        before = len(tctx.evidence)
        result = await self._registry.run(name, args, tctx)
        tool = self._registry.get(name)
        # A tool that does not number its own output (a custom tool, a skill)
        # gets one evidence number for the whole result, so it can be cited.
        if (
            result.status == "ok"
            and tool is not None
            and tool.citable
            and len(tctx.evidence) == before
            and result.content.strip()
        ):
            ev = tctx.evidence.add(tool=name, kind="tool_result", content=result.content)
            result.content = f"[{ev.n}] {result.content}"
        # The model sees a bounded result; the evidence keeps the full text.
        result.content = truncate(result.content, self._limits.max_observation_chars)
        return result

    def _calls_exhausted(self, tctx: ToolContext) -> bool:
        return int(tctx.state.get("tool_calls", 0)) >= self._limits.max_tool_calls

    def _record(
        self,
        trace: AgentTrace,
        observations: list[str],
        *,
        index: int,
        thought: str,
        action: str,
        action_input: dict[str, Any],
        result: ToolResult,
        call_id: str = "",
    ) -> None:
        if result.status == "ok":
            observations.append(result.content)
        cap = self._limits.max_trace_chars
        encoded = json.dumps(action_input, default=str)
        recorded_input = action_input if len(encoded) <= cap else {"_truncated": encoded[:cap]}
        trace.steps.append(
            AgentStep(
                index=index,
                thought=truncate(thought, cap),
                action=action,
                action_input=recorded_input,
                observation=truncate(result.content, cap),
                status=result.status,
                call_id=call_id,
            )
        )

    # ── Native tool calling ───────────────────────────────────────

    async def _run_native(
        self,
        query: str,
        history: list[ChatMessage],
        ctx: Context,
        tctx: ToolContext,
        trace: AgentTrace,
        observations: list[str],
    ) -> None:
        system = self._system_prompt or render_native_system_prompt()
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system),
            *history,
            ChatMessage(role="user", content=query),
        ]
        specs = self._registry.specs()
        step = 0
        try:
            for turn in range(self._max_steps):
                if ctx.budget_exceeded:
                    trace.stop_reason = "budget_exceeded"
                    ctx.log("Agentic loop stopped: latency budget exhausted")
                    return
                response = await self._llm.ainvoke_with_tools(
                    messages,
                    specs,
                    timeout=ctx.provider_timeout_seconds(f"agentic turn {turn}"),
                )
                calls = response.tool_calls or []
                if not calls:
                    trace.answer = (response.content or "").strip()
                    trace.stop_reason = "final_answer"
                    ctx.log(f"Agentic loop answered after {turn} tool turn(s)")
                    return
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=response.content or "",
                        tool_calls=[_echoable(c) for c in calls],
                    )
                )
                thought = (response.content or "").strip()
                for position, call in enumerate(calls):
                    if position >= self._limits.max_calls_per_turn:
                        result = refused(
                            call.name,
                            f"only {self._limits.max_calls_per_turn} tool calls run per turn; "
                            "call it again in the next turn if it is still needed",
                        )
                    else:
                        result = await self._native_call(call, tctx)
                    self._record(
                        trace,
                        observations,
                        index=step,
                        thought=thought,
                        action=call.name,
                        action_input=call.arguments,
                        result=result,
                        call_id=call.id,
                    )
                    step += 1
                    messages.append(
                        ChatMessage(role="tool", content=result.content, tool_call_id=call.id)
                    )
                if self._calls_exhausted(tctx):
                    trace.stop_reason = "max_tool_calls"
                    break
            else:
                trace.stop_reason = "max_steps"
            # Step or tool-call limit reached with work still pending: one last
            # turn without tools so the run ends with an answer, not silence.
            if ctx.budget_exceeded:
                return
            messages.append(ChatMessage(role="user", content=FINAL_ANSWER_NUDGE))
            response = await self._llm.ainvoke_with_tools(
                messages,
                specs,
                tool_choice="none",
                timeout=ctx.provider_timeout_seconds("agentic final answer"),
            )
            trace.answer = (response.content or "").strip()
        except LatencyBudgetExceededError:
            trace.stop_reason = "budget_exceeded"
            ctx.log("Agentic loop stopped: latency budget exhausted mid-run")

    async def _native_call(self, call: ToolCall, tctx: ToolContext) -> ToolResult:
        if call.parse_error:
            return refused(call.name, f"{call.parse_error}; send the arguments as a JSON object")
        return await self._call_tool(call.name, call.arguments, tctx)

    # ── Text ReAct fallback ───────────────────────────────────────

    async def _run_react(
        self,
        query: str,
        history: list[ChatMessage],
        ctx: Context,
        tctx: ToolContext,
        trace: AgentTrace,
        observations: list[str],
    ) -> None:
        system = render_system_prompt(
            tool_descriptions=self._registry.describe(),
            tool_names=", ".join(self._registry.names()),
        )
        if self._system_prompt:
            system = f"{self._system_prompt}\n\n{system}"
        conversation = "".join(f"{m.role.capitalize()}: {m.content}\n" for m in history)
        scratchpad = (f"Conversation so far:\n{conversation}\n" if conversation else "") + (
            f"Question: {query}\n"
        )
        reminded = False
        step = 0
        try:
            for turn in range(self._max_steps):
                if ctx.budget_exceeded:
                    trace.stop_reason = "budget_exceeded"
                    ctx.log("Agentic loop stopped: latency budget exhausted")
                    return
                prompt = f"{system}\n\n{scratchpad}\nThought:"
                response = await self._llm.ainvoke(
                    prompt, timeout=ctx.provider_timeout_seconds(f"agentic step {turn}")
                )
                parsed = parse_react_step(response.content or "")

                if "final_answer" in parsed:
                    trace.answer = parsed["final_answer"]
                    trace.stop_reason = "final_answer"
                    ctx.log(f"Agentic loop produced final answer at step {turn}")
                    return

                action = parsed.get("action", "")
                if not action:
                    if reminded:
                        trace.stop_reason = "no_action"
                        ctx.log("Agentic loop stopped: model emitted no action")
                        return
                    # One malformed reply gets a reminder of the format
                    # before the run gives up.
                    reminded = True
                    scratchpad += f"{REACT_FORMAT_REMINDER}\n"
                    continue

                action_input = parsed.get("action_input", {})
                result = await self._call_tool(action, action_input, tctx)
                self._record(
                    trace,
                    observations,
                    index=step,
                    thought=parsed.get("thought", ""),
                    action=action,
                    action_input=action_input,
                    result=result,
                )
                step += 1
                scratchpad += (
                    f"Thought: {parsed.get('thought', '')}\n"
                    f"Action: {action}\n"
                    f"Action Input: {json.dumps(action_input)}\n"
                    f"Observation: {result.content}\n"
                )
                if self._calls_exhausted(tctx):
                    trace.stop_reason = "max_tool_calls"
                    break
            else:
                trace.stop_reason = "max_steps"
            if ctx.budget_exceeded:
                return
            prompt = f"{system}\n\n{scratchpad}\n{FINAL_ANSWER_NUDGE}\nFinal Answer:"
            response = await self._llm.ainvoke(
                prompt, timeout=ctx.provider_timeout_seconds("agentic final answer")
            )
            text = response.content or ""
            parsed = parse_react_step(text)
            trace.answer = parsed.get("final_answer", text).strip()
        except LatencyBudgetExceededError:
            trace.stop_reason = "budget_exceeded"
            ctx.log("Agentic loop stopped: latency budget exhausted mid-run")

    def _format(self, raw: RawSearchResult) -> RetrieverResult:
        items = []
        for rec in raw.records:
            if isinstance(rec, dict):
                items.append(
                    RetrieverResultItem(
                        content=f"[{rec['n']}] {rec['content']}",
                        metadata={
                            "source": "agentic",
                            "citation": rec["n"],
                            "tool": rec.get("tool", ""),
                            "kind": rec.get("kind", ""),
                            "document": rec.get("source", ""),
                        },
                    )
                )
            else:
                items.append(RetrieverResultItem(content=str(rec), metadata={"source": "agentic"}))
        return RetrieverResult(items=items, metadata=raw.metadata)
