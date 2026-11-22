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


def test_a_missing_protobuf_import_asks_for_protobuf_not_gliner(monkeypatch):
    """gliner importing fine but protobuf missing at load must still name
    protobuf, and the model."""
    _broken_gliner(monkeypatch, ModuleNotFoundError("No module named 'google'", name="google"))
    monkeypatch.setattr(ee, "_protobuf_installed", lambda: False)
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    hint = info.value.hint
    assert "pip install protobuf" in hint and "pip install gliner" not in hint
    assert "urchade/gliner_medium-v2.1" in hint


def test_another_missing_import_is_named_not_blamed_on_gliner(monkeypatch):
    _broken_gliner(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    hint = info.value.hint
    assert "torch" in hint and "urchade/gliner_medium-v2.1" in hint
    assert "pip install gliner" not in hint


async def test_a_composite_stops_on_a_load_error_without_waiting_for_the_others():
    """A broken member's load error is raised while another member still
    runs; that member is cancelled, not awaited."""
    cancelled: list[str] = []

    class Broken(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            raise ModelLoadError("no model", hint="could not load")

    class Hangs(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(source_chunk_id)
                raise
            return []

    composite = ee.CompositeExtractor([Hangs(), Broken()])
    with pytest.raises(ModelLoadError):
        await asyncio.wait_for(composite.extract_entities("Alice", ["Person"], "c1"), timeout=5)
    assert cancelled == ["c1"]


async def test_another_members_error_does_not_hide_a_load_error():
    class NoFile(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            raise OSError("model file missing")

    class Broken(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            raise ModelLoadError("no model", hint="could not load")

    composite = ee.CompositeExtractor([NoFile(), Broken()])
    with pytest.raises(ModelLoadError):
        await composite.extract_entities("Alice", ["Person"], "c1")


def test_toolkit_logs_the_cause_the_hint_points_to(caplog):
    exc = ModelLoadError("Error parsing line in spm.model", hint="see the server log")
    with caplog.at_level("WARNING", logger="graphrag_sdk.tools.toolkit"):
        _file_error(exc)
    assert "Error parsing line in spm.model" in caplog.text


def test_toolkit_redacts_the_hint_in_the_log(caplog):
    exc = ModelLoadError("cause", hint="GLiNER model 'https://user:hunter2@host/m' could not load")
    with caplog.at_level("WARNING", logger="graphrag_sdk.tools.toolkit"):
        text = _file_error(exc)
    assert "hunter2" not in caplog.text and "hunter2" not in text


PROTOBUF, GLINER = "pip install protobuf", "pip install gliner"


@pytest.mark.parametrize(
    "exc, protobuf_installed, wants, rejects",
    [
        (ModuleNotFoundError("x", name="google.protobuf"), True, PROTOBUF, GLINER),
        (ModuleNotFoundError("x", name="google"), False, PROTOBUF, GLINER),
        (ModuleNotFoundError("x", name="google"), True, "'google'", "pip install protobuf"),
        (ModuleNotFoundError("x", name="google.cloud"), False, "'google.cloud'", "pip install"),
        (ImportError("cannot import name X"), False, PROTOBUF, GLINER),
        (ModuleNotFoundError("x", name="gliner"), True, "pip install gliner", "protobuf"),
    ],
    ids=[
        "protobuf",
        "bare google, no protobuf",
        "bare google, protobuf there",
        "other google",
        "unnamed, no protobuf",
        "gliner",
    ],
)
def test_import_failures_get_the_right_install_advice(
    monkeypatch, exc, protobuf_installed, wants, rejects
):
    _broken_gliner(monkeypatch, exc)
    monkeypatch.setattr(ee, "_protobuf_installed", lambda: protobuf_installed)
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    hint = info.value.hint
    assert wants in hint and rejects not in hint
    assert "urchade/gliner_medium-v2.1" in hint


async def test_cancelling_a_composite_is_not_replaced_by_a_members_load_error():
    release = asyncio.Event()

    class BrokenLater(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            await release.wait()
            raise ModelLoadError("no model", hint="could not load")

    class Hangs(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            await asyncio.Event().wait()
            return []

    composite = ee.CompositeExtractor([Hangs(), BrokenLater()])
    task = asyncio.ensure_future(composite.extract_entities("Alice", ["Person"], "c1"))
    for _ in range(3):
        await asyncio.sleep(0)
    release.set()
    task.cancel()
    await asyncio.wait([task])
    assert task.cancelled()


def test_a_hugging_face_token_in_the_cause_is_not_logged(caplog):
    token = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7"
    exc = ModelLoadError(f"401 for model, token {token}", hint="could not load")
    with caplog.at_level("WARNING", logger="graphrag_sdk.tools.toolkit"):
        _file_error(exc)
    assert token not in caplog.text and "hf_***" in caplog.text


def test_an_import_of_a_missing_name_is_a_version_problem_not_a_missing_package(monkeypatch):
    """``from transformers import X`` failing names transformers though it is
    installed: the advice is about versions, not installing it."""
    exc = ImportError("cannot import name 'X' from 'transformers'", name="transformers")
    _broken_gliner(monkeypatch, exc)
    monkeypatch.setattr(ee, "_protobuf_installed", lambda: True)
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    hint = info.value.hint
    assert "'transformers'" in hint and "version" in hint
    assert "not installed" not in hint


def test_a_plain_protobuf_import_error_is_a_version_problem(monkeypatch):
    """protobuf installed but missing a name asked of it: not a missing package."""
    exc = ImportError("cannot import name 'X' from 'google.protobuf'", name="google.protobuf")
    _broken_gliner(monkeypatch, exc)
    monkeypatch.setattr(ee, "_protobuf_installed", lambda: True)
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    hint = info.value.hint
    assert "'google.protobuf'" in hint and "version" in hint
    assert "not installed" not in hint


def test_an_import_failure_is_still_an_import_error_when_remembered(monkeypatch):
    """Callers that caught ImportError for a missing gliner still do, on the
    first failure and on the remembered one."""
    _broken_gliner(monkeypatch, ModuleNotFoundError("No module named 'gliner'", name="gliner"))
    for _ in range(2):
        with pytest.raises(ImportError) as info:
            ee.GLiNERExtractor()._load_model()
        assert isinstance(info.value, ModelLoadError)


def test_a_failure_that_is_not_an_import_is_not_an_import_error(monkeypatch):
    _broken_gliner(monkeypatch, OSError("offline"))
    with pytest.raises(ModelLoadError) as info:
        ee.GLiNERExtractor()._load_model()
    assert not isinstance(info.value, ImportError)
