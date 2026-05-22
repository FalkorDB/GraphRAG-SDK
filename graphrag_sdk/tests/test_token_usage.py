"""Tests for token usage tracking (#227).

Covers:
- TokenUsage model arithmetic and defaults
- Context.record_usage() accumulator
- LiteLLM / OpenRouter provider instrumentation
- Result types carry usage (IngestionResult, RagResult, RetrieverResult)
- Backward compatibility: ctx=None never raises
- abatch_invoke threads ctx through to each ainvoke
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    ChatMessage,
    IngestionResult,
    LLMResponse,
    RagResult,
    RetrieverResult,
    RetrieverResultItem,
    TokenUsage,
)
from graphrag_sdk.core.providers import LLMInterface, LiteLLM, LiteLLMEmbedder, OpenRouterLLM
from graphrag_sdk.core.providers.openrouter import OpenRouterEmbedder


# ── Helpers ──────────────────────────────────────────────────────


def _litellm_resp(content: str = "ok", prompt_tokens: int = 10, completion_tokens: int = 5):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _litellm_embed_resp(vectors: list[list[float]], prompt_tokens: int = 20):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    resp = MagicMock()
    resp.data = data
    resp.usage = usage
    return resp


def _openai_resp(content: str = "ok", prompt_tokens: int = 8, completion_tokens: int = 4):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _openai_embed_resp(vectors: list[list[float]], prompt_tokens: int = 15):
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    items = []
    for i, v in enumerate(vectors):
        item = MagicMock()
        item.index = i
        item.embedding = v
        items.append(item)
    resp = MagicMock()
    resp.data = items
    resp.usage = usage
    return resp


# ── TokenUsage model ─────────────────────────────────────────────


class TestTokenUsageModel:
    def test_defaults_are_zero(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.embedding_tokens == 0

    def test_explicit_values(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, embedding_tokens=200)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.embedding_tokens == 200

    def test_add_returns_new_instance(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, embedding_tokens=20)
        b = TokenUsage(prompt_tokens=3, completion_tokens=2, embedding_tokens=8)
        c = a + b
        assert c.prompt_tokens == 13
        assert c.completion_tokens == 7
        assert c.embedding_tokens == 28
        # originals unchanged
        assert a.prompt_tokens == 10
        assert b.prompt_tokens == 3

    def test_add_identity(self):
        a = TokenUsage(prompt_tokens=5)
        assert (a + TokenUsage()).prompt_tokens == 5

    def test_iadd_accumulates_in_place(self):
        u = TokenUsage()
        u += TokenUsage(prompt_tokens=10, completion_tokens=3, embedding_tokens=15)
        u += TokenUsage(prompt_tokens=5, completion_tokens=2, embedding_tokens=5)
        assert u.prompt_tokens == 15
        assert u.completion_tokens == 5
        assert u.embedding_tokens == 20

    def test_serializes_to_dict(self):
        u = TokenUsage(prompt_tokens=1, completion_tokens=2, embedding_tokens=3)
        d = u.model_dump()
        assert d == {"prompt_tokens": 1, "completion_tokens": 2, "embedding_tokens": 3}


# ── Context accumulator ──────────────────────────────────────────


class TestContextUsageAccumulator:
    def test_fresh_context_has_zero_usage(self):
        ctx = Context()
        assert ctx.usage.prompt_tokens == 0
        assert ctx.usage.completion_tokens == 0
        assert ctx.usage.embedding_tokens == 0

    def test_record_usage_llm(self):
        ctx = Context()
        ctx.record_usage(prompt_tokens=100, completion_tokens=50)
        assert ctx.usage.prompt_tokens == 100
        assert ctx.usage.completion_tokens == 50
        assert ctx.usage.embedding_tokens == 0

    def test_record_usage_embedding(self):
        ctx = Context()
        ctx.record_usage(embedding_tokens=300)
        assert ctx.usage.embedding_tokens == 300
        assert ctx.usage.prompt_tokens == 0

    def test_record_usage_accumulates(self):
        ctx = Context()
        ctx.record_usage(prompt_tokens=50, completion_tokens=10)
        ctx.record_usage(prompt_tokens=30, completion_tokens=5)
        ctx.record_usage(embedding_tokens=100)
        assert ctx.usage.prompt_tokens == 80
        assert ctx.usage.completion_tokens == 15
        assert ctx.usage.embedding_tokens == 100

    def test_record_usage_noop_zeros(self):
        ctx = Context()
        ctx.record_usage()
        assert ctx.usage.prompt_tokens == 0

    def test_usage_field_is_token_usage_instance(self):
        ctx = Context()
        assert isinstance(ctx.usage, TokenUsage)

    def test_two_contexts_have_independent_usage(self):
        ctx1 = Context()
        ctx2 = Context()
        ctx1.record_usage(prompt_tokens=999)
        assert ctx2.usage.prompt_tokens == 0

    def test_child_context_has_independent_usage(self):
        parent = Context()
        parent.record_usage(prompt_tokens=50)
        child = parent.child()
        child.record_usage(prompt_tokens=10)
        # parent still has original value
        assert parent.usage.prompt_tokens == 50
        assert child.usage.prompt_tokens == 10


# ── Result types carry usage ─────────────────────────────────────


class TestResultUsageField:
    def test_ingestion_result_default_usage(self):
        r = IngestionResult()
        assert isinstance(r.usage, TokenUsage)
        assert r.usage.prompt_tokens == 0

    def test_rag_result_default_usage(self):
        r = RagResult(answer="hello")
        assert isinstance(r.usage, TokenUsage)
        assert r.usage.embedding_tokens == 0

    def test_retriever_result_default_usage(self):
        r = RetrieverResult()
        assert isinstance(r.usage, TokenUsage)
        assert r.usage.completion_tokens == 0

    def test_ingestion_result_with_usage(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=40, embedding_tokens=200)
        r = IngestionResult(usage=u)
        assert r.usage.prompt_tokens == 100
        assert r.usage.embedding_tokens == 200

    def test_rag_result_with_usage(self):
        u = TokenUsage(prompt_tokens=50, completion_tokens=20)
        r = RagResult(answer="42", usage=u)
        assert r.usage.prompt_tokens == 50

    def test_retriever_result_with_usage(self):
        u = TokenUsage(embedding_tokens=75)
        r = RetrieverResult(items=[], usage=u)
        assert r.usage.embedding_tokens == 75

    def test_usage_mutability_on_retriever_result(self):
        """Facade sets usage after retrieval strategy returns."""
        r = RetrieverResult()
        ctx = Context()
        ctx.record_usage(embedding_tokens=42)
        r.usage = ctx.usage
        assert r.usage.embedding_tokens == 42

    def test_existing_fields_unchanged(self):
        """Backward compat: existing fields work exactly as before."""
        r = IngestionResult(nodes_created=5, relationships_created=3, chunks_indexed=2)
        assert r.nodes_created == 5
        assert r.relationships_created == 3
        assert r.chunks_indexed == 2

    def test_retriever_result_items_still_work(self):
        items = [RetrieverResultItem(content="a"), RetrieverResultItem(content="b")]
        r = RetrieverResult(items=items)
        assert len(r.items) == 2


# ── LiteLLM provider instrumentation ────────────────────────────


class TestLiteLLMUsageInstrumentation:
    @pytest.mark.asyncio
    async def test_ainvoke_records_usage_when_ctx_provided(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            return_value=_litellm_resp(prompt_tokens=30, completion_tokens=12)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            ctx = Context()
            await llm.ainvoke("hello", ctx=ctx)

        assert ctx.usage.prompt_tokens == 30
        assert ctx.usage.completion_tokens == 12
        assert ctx.usage.embedding_tokens == 0

    @pytest.mark.asyncio
    async def test_ainvoke_no_ctx_does_not_raise(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            return_value=_litellm_resp(prompt_tokens=10, completion_tokens=5)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            result = await llm.ainvoke("hello")  # no ctx → backward compat

        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_ainvoke_messages_records_usage(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            return_value=_litellm_resp(prompt_tokens=50, completion_tokens=20)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            ctx = Context()
            msgs = [ChatMessage(role="user", content="hi")]
            await llm.ainvoke_messages(msgs, ctx=ctx)

        assert ctx.usage.prompt_tokens == 50
        assert ctx.usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_ainvoke_messages_no_ctx_does_not_raise(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=_litellm_resp())
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            msgs = [ChatMessage(role="user", content="hi")]
            result = await llm.ainvoke_messages(msgs)

        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_aembed_query_records_embedding_tokens(self):
        mock_litellm = MagicMock()
        mock_litellm.aembedding = AsyncMock(
            return_value=_litellm_embed_resp([[0.1, 0.2]], prompt_tokens=25)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            ctx = Context()
            result = await embedder.aembed_query("hello", ctx=ctx)

        assert result == [0.1, 0.2]
        assert ctx.usage.embedding_tokens == 25
        assert ctx.usage.prompt_tokens == 0

    @pytest.mark.asyncio
    async def test_aembed_query_no_ctx_does_not_raise(self):
        mock_litellm = MagicMock()
        mock_litellm.aembedding = AsyncMock(
            return_value=_litellm_embed_resp([[0.1, 0.2]], prompt_tokens=25)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            result = await embedder.aembed_query("hello")  # no ctx

        assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_aembed_documents_records_embedding_tokens(self):
        """Regression: batch path (main ingest path) must accumulate embedding_tokens."""
        mock_litellm = MagicMock()
        mock_litellm.aembedding = AsyncMock(
            return_value=_litellm_embed_resp([[0.1, 0.2], [0.3, 0.4]], prompt_tokens=40)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            ctx = Context()
            results = await embedder.aembed_documents(["hello", "world"], ctx=ctx)

        assert len(results) == 2
        assert ctx.usage.embedding_tokens == 40

    @pytest.mark.asyncio
    async def test_aembed_documents_no_ctx_does_not_raise(self):
        mock_litellm = MagicMock()
        mock_litellm.aembedding = AsyncMock(
            return_value=_litellm_embed_resp([[0.1, 0.2], [0.3, 0.4]], prompt_tokens=40)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            embedder = LiteLLMEmbedder(model="text-embedding-ada-002")
            results = await embedder.aembed_documents(["hello", "world"])  # no ctx

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_multiple_ainvoke_calls_accumulate(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _litellm_resp(prompt_tokens=10, completion_tokens=5),
                _litellm_resp(prompt_tokens=20, completion_tokens=8),
            ]
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            ctx = Context()
            await llm.ainvoke("first", ctx=ctx)
            await llm.ainvoke("second", ctx=ctx)

        assert ctx.usage.prompt_tokens == 30
        assert ctx.usage.completion_tokens == 13

    @pytest.mark.asyncio
    async def test_usage_none_in_response_safe(self):
        """If litellm returns usage=None, record_usage gets zeros."""
        mock_litellm = MagicMock()
        resp = _litellm_resp()
        resp.usage = None
        mock_litellm.acompletion = AsyncMock(return_value=resp)
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            ctx = Context()
            await llm.ainvoke("hello", ctx=ctx)

        assert ctx.usage.prompt_tokens == 0
        assert ctx.usage.completion_tokens == 0

    @pytest.mark.asyncio
    async def test_astream_records_usage(self):
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            return_value=_litellm_resp(prompt_tokens=40, completion_tokens=20)
        )
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            llm = LiteLLM(model="gpt-4")
            ctx = Context()
            async for chunk in llm.astream("hello", ctx=ctx):
                pass
        
        assert ctx.usage.prompt_tokens == 40
        assert ctx.usage.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_abatch_invoke_accumulates_all_items(self):
        call_count = {"n": 0}

        class InstrumentedLLM(LLMInterface):
            def __init__(self):
                super().__init__(model_name="test")

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(content="ok")

            async def ainvoke(self, prompt: str, *, ctx=None, max_retries=3, **kwargs):
                call_count["n"] += 1
                if ctx is not None:
                    ctx.record_usage(prompt_tokens=10, completion_tokens=5)
                return LLMResponse(content="ok")

        llm = InstrumentedLLM()
        ctx = Context()
        results = await llm.abatch_invoke(["a", "b", "c"], ctx=ctx)

        assert len(results) == 3
        assert all(r.ok for r in results)
        assert ctx.usage.prompt_tokens == 30   # 3 × 10
        assert ctx.usage.completion_tokens == 15  # 3 × 5

    @pytest.mark.asyncio
    async def test_abatch_invoke_no_ctx_backward_compat(self):
        """abatch_invoke without ctx still works exactly as before."""
        class SimpleLLM(LLMInterface):
            def __init__(self):
                super().__init__(model_name="test")

            def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(content=f"resp:{prompt}")

        llm = SimpleLLM()
        results = await llm.abatch_invoke(["x", "y"])

        assert len(results) == 2
        assert results[0].ok


# ── OpenRouter provider instrumentation ──────────────────────────


class TestOpenRouterUsageInstrumentation:
    @pytest.mark.asyncio
    async def test_ainvoke_records_usage(self):
        mock_openai = MagicMock()
        mock_async = MagicMock()
        mock_async.chat.completions.create = AsyncMock(
            return_value=_openai_resp(prompt_tokens=8, completion_tokens=4)
        )
        mock_openai.AsyncOpenAI.return_value = mock_async
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="openai/gpt-4o", api_key="k")
            ctx = Context()
            result = await llm.ainvoke("hello", ctx=ctx)

        assert result.content == "ok"
        assert ctx.usage.prompt_tokens == 8
        assert ctx.usage.completion_tokens == 4

    @pytest.mark.asyncio
    async def test_ainvoke_no_ctx_backward_compat(self):
        mock_openai = MagicMock()
        mock_async = MagicMock()
        mock_async.chat.completions.create = AsyncMock(return_value=_openai_resp())
        mock_openai.AsyncOpenAI.return_value = mock_async
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="openai/gpt-4o", api_key="k")
            result = await llm.ainvoke("hello")

        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_ainvoke_messages_records_usage(self):
        mock_openai = MagicMock()
        mock_async = MagicMock()
        mock_async.chat.completions.create = AsyncMock(
            return_value=_openai_resp(prompt_tokens=12, completion_tokens=6)
        )
        mock_openai.AsyncOpenAI.return_value = mock_async
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="openai/gpt-4o", api_key="k")
            ctx = Context()
            msgs = [ChatMessage(role="user", content="hi")]
            await llm.ainvoke_messages(msgs, ctx=ctx)

        assert ctx.usage.prompt_tokens == 12
        assert ctx.usage.completion_tokens == 6

    @pytest.mark.asyncio
    async def test_aembed_query_records_embedding_tokens(self):
        mock_openai = MagicMock()
        mock_async = MagicMock()
        mock_async.embeddings.create = AsyncMock(
            return_value=_openai_embed_resp([[0.3, 0.4]], prompt_tokens=18)
        )
        mock_openai.AsyncOpenAI.return_value = mock_async
        with patch.dict("sys.modules", {"openai": mock_openai}):
            embedder = OpenRouterEmbedder(model="text-embedding-ada-002", api_key="k")
            ctx = Context()
            result = await embedder.aembed_query("hello", ctx=ctx)

        assert result == [0.3, 0.4]
        assert ctx.usage.embedding_tokens == 18

    @pytest.mark.asyncio
    async def test_aembed_query_no_ctx_backward_compat(self):
        mock_openai = MagicMock()
        mock_async = MagicMock()
        mock_async.embeddings.create = AsyncMock(
            return_value=_openai_embed_resp([[0.3, 0.4]])
        )
        mock_openai.AsyncOpenAI.return_value = mock_async
        with patch.dict("sys.modules", {"openai": mock_openai}):
            embedder = OpenRouterEmbedder(model="text-embedding-ada-002", api_key="k")
            result = await embedder.aembed_query("hello")  # no ctx

        assert result == [0.3, 0.4]

    @pytest.mark.asyncio
    async def test_usage_none_in_response_safe(self):
        mock_openai = MagicMock()
        mock_async = MagicMock()
        resp = _openai_resp()
        resp.usage = None
        mock_async.chat.completions.create = AsyncMock(return_value=resp)
        mock_openai.AsyncOpenAI.return_value = mock_async
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = OpenRouterLLM(model="openai/gpt-4o", api_key="k")
            ctx = Context()
            await llm.ainvoke("hello", ctx=ctx)

        assert ctx.usage.prompt_tokens == 0
        assert ctx.usage.completion_tokens == 0


# ── Usage extraction helpers ──────────────────────────────────────


class TestUsageExtractionHelpers:
    def test_litellm_extract_llm_usage_normal(self):
        from graphrag_sdk.core.providers.litellm import _extract_llm_usage
        resp = _litellm_resp(prompt_tokens=42, completion_tokens=17)
        pt, ct = _extract_llm_usage(resp)
        assert pt == 42
        assert ct == 17

    def test_litellm_extract_llm_usage_none(self):
        from graphrag_sdk.core.providers.litellm import _extract_llm_usage
        resp = MagicMock()
        resp.usage = None
        pt, ct = _extract_llm_usage(resp)
        assert pt == 0
        assert ct == 0

    def test_litellm_extract_embedding_usage_normal(self):
        from graphrag_sdk.core.providers.litellm import _extract_embedding_usage
        resp = _litellm_embed_resp([[0.1]], prompt_tokens=33)
        assert _extract_embedding_usage(resp) == 33

    def test_litellm_extract_embedding_usage_none(self):
        from graphrag_sdk.core.providers.litellm import _extract_embedding_usage
        resp = MagicMock()
        resp.usage = None
        assert _extract_embedding_usage(resp) == 0

    def test_openrouter_extract_llm_usage_normal(self):
        from graphrag_sdk.core.providers.openrouter import _extract_openai_llm_usage
        resp = _openai_resp(prompt_tokens=99, completion_tokens=11)
        pt, ct = _extract_openai_llm_usage(resp)
        assert pt == 99
        assert ct == 11

    def test_openrouter_extract_embedding_usage_normal(self):
        from graphrag_sdk.core.providers.openrouter import _extract_openai_embedding_usage
        resp = _openai_embed_resp([[0.1]], prompt_tokens=77)
        assert _extract_openai_embedding_usage(resp) == 77

    def test_openrouter_extract_with_missing_attribute(self):
        from graphrag_sdk.core.providers.openrouter import _extract_openai_llm_usage
        resp = MagicMock(spec=[])  # no attributes
        pt, ct = _extract_openai_llm_usage(resp)
        assert pt == 0
        assert ct == 0


# ── Public export ─────────────────────────────────────────────────


class TestPublicExport:
    def test_token_usage_importable_from_top_level(self):
        from graphrag_sdk import TokenUsage  # noqa: F401
        assert TokenUsage is not None

    def test_token_usage_in_all(self):
        import graphrag_sdk
        assert "TokenUsage" in graphrag_sdk.__all__

    def test_ingestion_result_in_all(self):
        import graphrag_sdk
        assert "IngestionResult" in graphrag_sdk.__all__
        assert "RagResult" in graphrag_sdk.__all__
        assert "RetrieverResult" in graphrag_sdk.__all__


# ── End-to-end accumulation scenario (no real LLM) ───────────────


class TestEndToEndAccumulation:
    @pytest.mark.asyncio
    async def test_mixed_llm_and_embed_accumulate_in_single_ctx(self):
        """Simulate a retrieve+complete flow accumulating into one ctx."""
        ctx = Context()

        # Step 1: embed query (simulate retrieval)
        ctx.record_usage(embedding_tokens=50)

        # Step 2: keyword extraction LLM call
        ctx.record_usage(prompt_tokens=200, completion_tokens=10)

        # Step 3: final answer LLM call
        ctx.record_usage(prompt_tokens=1500, completion_tokens=300)

        assert ctx.usage.prompt_tokens == 1700
        assert ctx.usage.completion_tokens == 310
        assert ctx.usage.embedding_tokens == 50

        # Snapshot into RagResult
        result = RagResult(answer="The answer", usage=ctx.usage)
        assert result.usage.prompt_tokens == 1700
        assert result.usage.completion_tokens == 310
        assert result.usage.embedding_tokens == 50

    @pytest.mark.asyncio
    async def test_ingest_flow_accumulation(self):
        """Simulate ingest: chunking LLM + extraction LLM + embedding."""
        ctx = Context()

        # ContextualChunking: 3 chunks × ~100 tokens each
        ctx.record_usage(prompt_tokens=300, completion_tokens=90)

        # GraphExtraction step1 NER + step2 verify
        ctx.record_usage(prompt_tokens=2000, completion_tokens=800)

        # Embedding 3 chunks
        ctx.record_usage(embedding_tokens=150)

        result = IngestionResult(
            nodes_created=5,
            chunks_indexed=3,
            usage=ctx.usage,
        )
        assert result.usage.prompt_tokens == 2300
        assert result.usage.completion_tokens == 890
        assert result.usage.embedding_tokens == 150
        # original fields unchanged
        assert result.nodes_created == 5
        assert result.chunks_indexed == 3
