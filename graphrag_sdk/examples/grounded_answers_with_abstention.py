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
your corpus genuinely does not cover, you need per-item scores.

``main()`` uses ``LocalRetrieval`` for that, because its items are individual
chunks and ``score`` is a real per-chunk similarity. Under
``MultiPathRetrieval`` each item is a whole concatenated section instead, so a
``CosineReranker`` scores the entire blob against the question — a diluted,
section-level number. That still works, but the threshold has to be calibrated
against measured section scores; do not carry ``MIN_SCORE`` across.

``answer_or_abstain()`` below is deliberately small and dependency-free so
you can copy it into your own application (it is also exercised by
``graphrag_sdk/tests/test_grounded_abstention.py``).

Cost note: the gate retrieves once, and ``completion()`` retrieves again
internally, so the retrieval work always happens twice. What that costs
depends entirely on the strategy. ``LocalRetrieval``, used by ``main()``,
makes no LLM calls at all: a refused question is genuinely free, and an
answered one costs exactly the one generation call. ``MultiPathRetrieval``
runs an LLM keyword-extraction call on *every* retrieval, so there the gate
is not free — a refused question still costs one call, and an answered one
costs three instead of two.

Either way the trade is worth it when a meaningful share of your traffic is
unanswerable; if almost every question is answerable, gate on a cheaper
signal or accept the prompt-level abstention in rule 6 of the default system
prompt.

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
    FalkorDBConnection,
    GraphRAG,
    GraphStore,
    LiteLLM,
    LiteLLMEmbedder,
    VectorStore,
)
from graphrag_sdk.retrieval.strategies.local import LocalRetrieval

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
            ``history=`` or ``strategy=``). ``rewrite_question_with_history``
            is rejected — pass an already-resolved question instead.

    Raises:
        ValueError: If ``min_items`` is negative, or if
            ``rewrite_question_with_history=True`` is passed.

    Returns:
        A ``GroundedAnswer``. When ``grounded`` is False the answer is the
        refusal string and no generation happened.
    """
    if min_items < 0:
        raise ValueError("min_items must be non-negative")

    # A follow-up question has to be resolved *before* the gate runs —
    # retrieving on a bare "What did she build?" finds nothing useful. But
    # ``completion()`` rewrites internally, after its own retrieval, so there
    # is no public hook to reuse here. Rather than reach into private methods
    # (this file gets copied into applications), require the caller to pass
    # text that already stands on its own.
    if completion_kwargs.pop("rewrite_question_with_history", False):
        raise ValueError(
            "rewrite_question_with_history=True is not supported by this gate: "
            "the gate must retrieve on a self-contained question, and the SDK "
            "resolves follow-ups only inside completion(), after retrieval. "
            "Resolve the question yourself and pass the resolved text as "
            "`question`."
        )

    retrieval_kwargs = {
        key: completion_kwargs[key]
        for key in ("strategy", "reranker", "ctx")
        if key in completion_kwargs
    }
    retrieved = await rag.retrieve(question, **retrieval_kwargs)
    supporting = [item for item in retrieved.items if is_evidence(item, min_score)]

    if len(supporting) < min_items:
        return GroundedAnswer(question=question, answer=refusal, grounded=False)

    completion_kwargs["return_context"] = True

    # Don't hand the same Context to generation. ``Context`` measures its
    # budget as time-since-construction, so a ctx already spent on the gate
    # arrives at generation with a silently shorter deadline — and the gate is
    # the cheap half. ``child()`` gives generation a fresh start time while
    # carrying over only the budget that actually remains, so the two-phase
    # sequence stays inside the caller's overall cap without the second call
    # inheriting the first call's elapsed time.
    gate_ctx = completion_kwargs.get("ctx")
    if gate_ctx is not None:
        completion_kwargs["ctx"] = gate_ctx.child()

    result = await rag.completion(question, **completion_kwargs)

    # ``completion()`` runs its own retrieval, so the two passes can disagree.
    # Cite only what generation actually retrieved: substituting the gate's
    # items would attach provenance the answer never saw, which is the exact
    # failure this example exists to prevent. If generation's own evidence
    # fails the same bar, refuse — a passing gate does not license an
    # ungrounded answer.
    generated_from = result.retriever_result.items if result.retriever_result else []
    cited = [item for item in generated_from if is_evidence(item, min_score)]

    if len(cited) < min_items:
        return GroundedAnswer(question=question, answer=refusal, grounded=False)

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

    connection = FalkorDBConnection(
        ConnectionConfig(host="localhost", graph_name="grounded_abstention")
    )

    # Why LocalRetrieval here, and not the default MultiPathRetrieval:
    # MultiPathRetrieval emits one item per *section*, so a reranker scores a
    # whole concatenated markdown blob (up to 15 passages joined together)
    # against the question. That similarity is diluted by everything else in
    # the blob, which makes MIN_SCORE a section-level quantity no reader can
    # predict. LocalRetrieval items are individual chunks, so a similarity
    # floor means exactly what it says and the demo behaves the same way on
    # your machine as it does here.
    #
    # The trade-off is real: you lose the graph-traversal paths. Use
    # MultiPathRetrieval when you need them, but calibrate MIN_SCORE against
    # measured section scores rather than reusing this value.
    scored_strategy = LocalRetrieval(
        graph_store=GraphStore(connection),
        vector_store=VectorStore(connection, embedder=embedder, embedding_dimension=256),
        embedder=embedder,
        top_k=5,
    )

    async with GraphRAG(
        connection=connection,
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
                strategy=scored_strategy,
            )
        )

        # ── 3. A question the corpus says nothing about → abstain ──
        # The corpus covers Acme's founding, offices and staff, but says
        # nothing about revenue. Vector search still returns the Acme chunks —
        # they share vocabulary with the question — so the abstention depends
        # on those chunks scoring below MIN_SCORE, not on retrieval coming
        # back empty. Because these are per-chunk similarities, that is a
        # quantity you can inspect: print the scores first (see the "Are the
        # scores plausible?" check in the Reliability and Grounding docs),
        # then set the floor just under the weakest answer you would accept.
        report(
            await answer_or_abstain(
                rag,
                "What was Acme Corp's revenue in 2024?",
                min_score=MIN_SCORE,
                strategy=scored_strategy,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
