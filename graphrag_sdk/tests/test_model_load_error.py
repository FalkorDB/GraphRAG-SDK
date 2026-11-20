"""A model that cannot load: a safe hint reaches the user, and extraction stops promptly."""

from __future__ import annotations

import asyncio
import sys

import pytest

from graphrag_sdk.core.exceptions import ModelLoadError
from graphrag_sdk.core.models import Ontology, TextChunk, TextChunks
from graphrag_sdk.ingestion.extraction_strategies import entity_extractors as ee
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import EntityExtractor
from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import GraphExtraction
from graphrag_sdk.mcp.server import describe_error
from graphrag_sdk.tools.toolkit import _file_error

from .conftest import MockLLMWithGraphExtraction

SECRET = "sk-abcdefghijklmnopqrstuvwxyz0123456789"


def _broken_gliner(monkeypatch, exc):
    class Broken:
        @staticmethod
        def from_pretrained(name):
            raise exc

    monkeypatch.setitem(sys.modules, "gliner", type("m", (), {"GLiNER": Broken}))
    monkeypatch.setattr(ee.GLiNERExtractor, "_MODEL_CACHE", {})
    monkeypatch.setattr(ee.GLiNERExtractor, "_LOAD_ERRORS", {})


def test_the_load_error_carries_a_hint_without_the_underlying_text(monkeypatch):
    _broken_gliner(monkeypatch, ValueError(f"Error parsing line, token={SECRET}"))
    monkeypatch.setattr(ee, "_protobuf_installed", lambda: False)
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    hint = info.value.hint
    assert "urchade/gliner_medium-v2.1" in hint and "pip install protobuf" in hint
    assert SECRET not in hint


def test_the_hint_survives_the_remembered_failure(monkeypatch):
    _broken_gliner(monkeypatch, OSError("offline"))
    for _ in range(2):
        with pytest.raises(ModelLoadError) as info:
            ee.GLiNERExtractor()._load_model()
        assert "could not load" in info.value.hint


def test_toolkit_file_error_shows_the_hint_not_the_message():
    exc = ModelLoadError(f"detail {SECRET}", hint="Install it with: pip install protobuf")
    text = _file_error(exc)
    assert "pip install protobuf" in text and SECRET not in text


def test_mcp_error_shows_the_hint_not_the_message():
    exc = ModelLoadError(f"detail {SECRET}", hint="Install it with: pip install protobuf")
    text = describe_error("ingest", exc)
    assert "pip install protobuf" in text and SECRET not in text
    assert "could not load" in text


async def test_extraction_stops_on_the_first_load_error_without_waiting(ctx):
    """The first chunk's load error is raised while other chunks still hang;
    the hanging chunks are cancelled, not awaited."""
    cancelled: list[str] = []

    class FirstFailsRestHang(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            if text.startswith("first"):
                raise ModelLoadError("no model", hint="could not load")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(source_chunk_id)
                raise
            return []

    chunks = TextChunks(
        chunks=[
            TextChunk(text=t, index=i, uid=f"c{i}")
            for i, t in enumerate(["first chunk", "second chunk", "third chunk"])
        ]
    )
    extractor = GraphExtraction(
        llm=MockLLMWithGraphExtraction(), entity_extractor=FirstFailsRestHang()
    )
    with pytest.raises(ModelLoadError):
        await asyncio.wait_for(extractor.extract(chunks, Ontology(), ctx), timeout=5)
    assert sorted(cancelled) == ["c1", "c2"]
