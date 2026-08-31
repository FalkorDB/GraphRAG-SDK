"""
GraphRAG SDK -- Grounded Answers with Abstention
===================================================
Reduce LLM hallucinations by answering only from retrieved graph context,
and abstaining explicitly when the graph holds no supporting evidence.

The pattern has two halves:

  1. Gate on retrieval — call ``rag.retrieve(question)`` first. If nothing
     comes back (or everything is below your score threshold), return an
     "evidence-insufficient" response instead of paying for a generation
     the model would have to guess at.
  2. Ground the answer — when evidence exists, call
     ``rag.completion(question, return_context=True)`` so the retrieval
     trail comes back alongside the answer and every claim can be traced
     to its source chunks.

Counting items is not enough on its own. No path in the default
``MultiPathRetrieval`` strategy applies a similarity floor — chunk vector
search takes the top 15 hits and reranking takes the top *k* of those, both
without a threshold — so a non-empty graph nearly always returns *something*,
however unrelated. A count-only gate therefore fires only in the degenerate
cases (empty graph, no ingestion, retrieval error). To abstain on a question
your corpus genuinely does not cover, you need per-item scores: attach a
``CosineReranker`` (or use ``LocalRetrieval``) and pass ``min_score``, as
``main()`` does below.

``answer_or_abstain()`` below is deliberately small and dependency-free so
you can copy it into your own application (it is also exercised by
``graphrag_sdk/tests/test_grounded_abstention.py``).

Cost note: the gate retrieves once, and ``completion()`` retrieves again
internally. Because ``MultiPathRetrieval`` runs an LLM keyword-extraction
call on every retrieval, an *answered* question costs three LLM calls
instead of two (four instead of three with
``rewrite_question_with_history=True``), and does the graph and vector work
twice. An *unanswered* question still costs the retrieval keyword-extraction
LLM call (plus the rewrite call when enabled), but avoids the generation call.
That trade is worth it when a meaningful share of your traffic is unanswerable;
if almost every question is answerable, gate on a cheaper signal or accept the
prompt-level abstention in rule 6 of the default system prompt.

Prerequisites:
    docker run -p 6379:6379 falkordb/falkordb
    pip install graphrag-sdk[litellm]
    export OPENAI_API_KEY="sk-..."

Run:
    python graphrag_sdk/examples/grounded_answers_with_abstention.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from graphrag_sdk import (
    ConnectionConfig,
    Context,
    CosineReranker,
    GraphRAG,
    LiteLLM,
    LiteLLMEmbedder,
)

TEXT = (
    "Acme Corp was founded in 2015 by Alice Johnson and is headquartered in London. "
    "Alice Johnson is the CTO and leads the backend team. "
    "Acme Corp builds cloud infrastructure for logistics companies. "
    "In 2026 Acme Corp opened a second office in Berlin, managed by Clara Wei."
)

#: Returned verbatim when the graph holds no supporting evidence. Make this
#: string explicit — a caller (or an eval harness) must be able to tell an
#: abstention apart from a real answer without parsing prose.
INSUFFICIENT_EVIDENCE = (
    "I don't have enough evidence in the knowledge graph to answer that question."
)

#: MultiPathRetrieval emits an answer-format hint section that carries no
#: document evidence. Ignore it when deciding whether to abstain.
NON_EVIDENCE_SECTIONS = frozenset({"hint"})

#: Similarity floor for a reranked item to count as evidence. Corpus- and
#: embedding-model-dependent: measure your own scores before trusting it.
MIN_SCORE = 0.35


@dataclass
class GroundedAnswer:
    """An answer plus the provenance that justifies it.

    ``grounded`` is False exactly when the abstention gate fired, in which
    case ``answer`` is the ``refusal`` string and no generation call was made.
    """

    question: str
    answer: str
    grounded: bool
    citations: list[str] = field(default_factory=list)
    context: list[Any] = field(default_factory=list)


def is_evidence(item: Any, min_score: float | None) -> bool:
    """Decide whether one retrieved item counts as supporting evidence.

    Items are dropped when they are empty, when they belong to a
    non-evidence section, or when a ``min_score`` threshold is set and the
    item scores below it. Items without a score are kept only when no
    threshold is requested — the default ``MultiPathRetrieval`` strategy
    does not populate ``score``, so pass ``min_score`` only with a scoring
    strategy (``LocalRetrieval``) or a reranker (``CosineReranker``).
    """
    if not (item.content or "").strip():
        return False
    if item.metadata.get("section") in NON_EVIDENCE_SECTIONS:
        return False
    if min_score is None:
        return True
    return item.score is not None and item.score >= min_score


def citation_of(item: Any, position: int) -> str:
    """Best-effort provenance label for a retrieved item."""
    metadata = item.metadata or {}
    for key in ("chunk_id", "document_id", "source", "section"):
        value = metadata.get(key)
        if value:
            return f"{key}={value}"
    return f"item[{position}]"


async def answer_or_abstain(
    rag: GraphRAG,
    question: str,
    *,
    min_items: int = 1,
    min_score: float | None = None,
    refusal: str = INSUFFICIENT_EVIDENCE,
    **completion_kwargs: Any,
) -> GroundedAnswer:
    """Answer from graph context, or abstain when the evidence is too thin.

    Args:
        rag: An initialized ``GraphRAG`` instance.
        question: The user's question.
        min_items: Minimum number of supporting context items required
            before the LLM is allowed to answer.
        min_score: Optional minimum retrieval score per item.
        refusal: Response returned when the gate fires.
        **completion_kwargs: Forwarded to ``rag.completion()`` (for example
            ``history=`` or ``strategy=``).

    Returns:
        A ``GroundedAnswer``. When ``grounded`` is False the answer is the
        refusal string and no generation happened.
    """
    if min_items < 0:
        raise ValueError("min_items must be non-negative")

    resolved_question = question
    rewrite_question = completion_kwargs.pop("rewrite_question_with_history", False)
    history = completion_kwargs.get("history")
    if rewrite_question and history:
        ctx = completion_kwargs.get("ctx")
        if ctx is None:
            ctx = Context()
            completion_kwargs["ctx"] = ctx
        validated_history = rag._validate_history(history)
        resolved_question = await rag._rewrite_question_with_history(
            question,
            validated_history,
            ctx=ctx,
        )

    retrieval_kwargs = {
        key: completion_kwargs[key]
        for key in ("strategy", "reranker", "ctx")
        if key in completion_kwargs
    }
    retrieved = await rag.retrieve(resolved_question, **retrieval_kwargs)
    supporting = [item for item in retrieved.items if is_evidence(item, min_score)]

    if len(supporting) < min_items:
        return GroundedAnswer(question=question, answer=refusal, grounded=False)

    completion_kwargs["return_context"] = True
    completion_kwargs["rewrite_question_with_history"] = False
    result = await rag.completion(resolved_question, **completion_kwargs)

    # ``return_context=True`` populates ``retriever_result``; fall back to the
    # gate's own retrieval if a custom pipeline leaves it unset.
    items = result.retriever_result.items if result.retriever_result else supporting
    cited = [item for item in items if is_evidence(item, min_score)]

    # ``completion()`` retrieves independently of the gate above, and keyword
    # extraction is an LLM call — the two passes can disagree. Never report a
    # grounded answer with no provenance; fall back to the evidence the gate
    # actually approved.
    if not cited:
        cited = supporting

    return GroundedAnswer(
        question=question,
        answer=result.answer,
        grounded=True,
        citations=[citation_of(item, i) for i, item in enumerate(cited)],
        context=cited,
    )


def report(result: GroundedAnswer) -> None:
    """Print an answer together with the evidence that backs it."""
    print(f"\nQ: {result.question}")
    print(f"A: {result.answer}")
    print(f"   grounded={result.grounded}")
    for citation, item in zip(result.citations, result.context):
        snippet = " ".join(item.content.split())[:160]
        print(f"   ↳ [{citation}] {snippet}")


async def main():
    llm = LiteLLM(model="openai/gpt-5.5")
    embedder = LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256)

    # Scores are what make the gate discriminating. Without a reranker the
    # default strategy leaves ``score`` unset on every item, and the gate
    # degrades to "did retrieval return anything at all?".
    reranker = CosineReranker(embedder=embedder, top_k=10)

    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="grounded_abstention"),
        llm=llm,
        embedder=embedder,
        embedding_dimension=256,
    ) as rag:
        # ── 1. Ingest a small, self-contained corpus ───────────────
        await rag.ingest(text=TEXT, document_id="acme_facts")
        await rag.finalize()

        # ── 2. A question the corpus supports → grounded answer ────
        report(
            await answer_or_abstain(
                rag,
                "Who founded Acme Corp?",
                min_score=MIN_SCORE,
                reranker=reranker,
            )
        )

        # ── 3. A question the corpus says nothing about → abstain ──
        # The corpus covers Acme's founding, offices and staff, but says
        # nothing about revenue. Retrieval still returns the Acme chunks —
        # they share vocabulary with the question — so the abstention
        # depends on those chunks scoring below MIN_SCORE, not on retrieval
        # coming back empty. Tune MIN_SCORE against your own corpus: print
        # the scores first (see the "Are the scores plausible?" check in the
        # Reliability and Grounding docs), then set the floor just under the
        # weakest answer you would still accept.
        report(
            await answer_or_abstain(
                rag,
                "What was Acme Corp's revenue in 2024?",
                min_score=MIN_SCORE,
                reranker=reranker,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
