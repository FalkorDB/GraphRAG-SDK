"""Tests for retrieval/strategies/chunk_retrieval.py utilities."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.retrieval.strategies.chunk_retrieval import fetch_chunk_documents


def _graph_with_rows(rows: list[list]) -> MagicMock:
    """Build a graph_store mock whose query_raw returns the given rows."""
    result = MagicMock()
    result.result_set = rows
    graph = MagicMock()
    graph.query_raw = AsyncMock(return_value=result)
    return graph


class TestFetchChunkDocuments:
    """fetch_chunk_documents returns the full Document.path verbatim.

    The path is the value passed to ``rag.ingest()`` as the source id
    (typically a path relative to the ingestion root). Downstream
    consumers — citation rendering, source-link builders — rely on the
    full relative path to disambiguate files that share a basename
    (e.g. ``operations/index.md`` vs ``commands/index.md``).
    """

    async def test_returns_full_relative_path_unchanged(self):
        graph = _graph_with_rows(
            [
                ["chunk-1", "operations/falkordblite/falkordblite-py.md"],
                ["chunk-2", "cypher/functions.md"],
                ["chunk-3", "genai-tools/graphrag-toolkit.md"],
            ]
        )
        mapping = await fetch_chunk_documents(graph, ["chunk-1", "chunk-2", "chunk-3"])

        assert mapping == {
            "chunk-1": "operations/falkordblite/falkordblite-py.md",
            "chunk-2": "cypher/functions.md",
            "chunk-3": "genai-tools/graphrag-toolkit.md",
        }

    async def test_preserves_paths_that_share_a_basename(self):
        # Three different files with the same basename — we must not
        # collapse them to identical mappings.
        graph = _graph_with_rows(
            [
                ["c-root", "index.md"],
                ["c-ops", "operations/index.md"],
                ["c-cmd", "commands/index.md"],
            ]
        )
        mapping = await fetch_chunk_documents(graph, ["c-root", "c-ops", "c-cmd"])

        assert mapping == {
            "c-root": "index.md",
            "c-ops": "operations/index.md",
            "c-cmd": "commands/index.md",
        }

    async def test_returns_basename_when_path_has_no_directory(self):
        graph = _graph_with_rows([["chunk-1", "configuration.md"]])
        mapping = await fetch_chunk_documents(graph, ["chunk-1"])
        assert mapping == {"chunk-1": "configuration.md"}

    async def test_skips_rows_with_empty_path(self):
        # Older ingests may have left Document.path blank — skip those.
        graph = _graph_with_rows(
            [
                ["chunk-1", ""],
                ["chunk-2", None],
                ["chunk-3", "cypher/match.md"],
            ]
        )
        mapping = await fetch_chunk_documents(graph, ["chunk-1", "chunk-2", "chunk-3"])
        assert mapping == {"chunk-3": "cypher/match.md"}

    async def test_empty_chunk_ids_short_circuits(self):
        graph = MagicMock()
        graph.query_raw = AsyncMock()
        mapping = await fetch_chunk_documents(graph, [])
        assert mapping == {}
        graph.query_raw.assert_not_awaited()

    async def test_query_failure_returns_empty_mapping(self):
        graph = MagicMock()
        graph.query_raw = AsyncMock(side_effect=RuntimeError("graph down"))
        mapping = await fetch_chunk_documents(graph, ["chunk-1"])
        assert mapping == {}
