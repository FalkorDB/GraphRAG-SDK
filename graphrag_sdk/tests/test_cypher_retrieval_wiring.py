"""Reaching the text-to-Cypher path, and reaching it with a current ontology.

Declaring column types only pays off if something reads them. Two things used to
stand in the way: the option was not on the client at all, so the documented way
to get an aggregate was to hand-build a strategy out of private attributes; and
the default strategy captured the ontology that existed when it was constructed,
which for a structured source is always *before* the mapping declared anything.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import Attribute, Entity, Ontology
from graphrag_sdk.retrieval.strategies.base import RetrievalStrategy
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval


@pytest.fixture
def mock_conn():
    conn = MagicMock(spec=FalkorDBConnection)
    result_mock = MagicMock()
    result_mock.result_set = []
    conn.query = AsyncMock(return_value=result_mock)
    conn.config = ConnectionConfig()
    ontology_graph = MagicMock()
    ontology_graph.query = AsyncMock(return_value=result_mock)
    conn._driver = MagicMock()
    conn._driver.select_graph = MagicMock(return_value=ontology_graph)
    conn._ensure_client = MagicMock()
    return conn


@pytest.fixture
def graphrag_factory(mock_conn, embedder, llm):
    def _make(**kwargs):
        return GraphRAG(
            connection=mock_conn,
            llm=llm,
            embedder=embedder,
            embedding_dimension=8,
            **kwargs,
        )

    return _make


@pytest.fixture
def graphrag(graphrag_factory):
    return graphrag_factory()


class TestTheOptionIsOnTheClient:
    def test_off_by_default(self, graphrag):
        assert graphrag._retrieval_strategy._enable_cypher is False

    def test_enabled_when_asked(self, graphrag_factory):
        rag = graphrag_factory(enable_cypher=True)
        assert rag._retrieval_strategy._enable_cypher is True

    def test_a_supplied_strategy_decides_for_itself(self, graphrag_factory):
        """The flag configures the default strategy. A caller who brings their
        own has already made the choice, and silently overriding it would be
        worse than ignoring the flag."""

        class Custom(RetrievalStrategy):
            async def _execute(self, query, ctx, **kwargs):  # pragma: no cover
                raise AssertionError("not called")

        supplied = Custom()
        rag = graphrag_factory(enable_cypher=True, retrieval_strategy=supplied)
        assert rag._retrieval_strategy is supplied


class TestTheStrategySeesTheCurrentOntology:
    def test_the_base_hook_is_a_no_op_not_an_error(self):
        """A strategy that ignores the ontology must not have to implement this."""

        class Minimal(RetrievalStrategy):
            async def _execute(self, query, ctx, **kwargs):  # pragma: no cover
                raise AssertionError("not called")

        Minimal().set_ontology(Ontology())  # must not raise

    def test_multi_path_adopts_a_new_ontology(self):
        strategy = MultiPathRetrieval(
            graph_store=object(),
            vector_store=object(),
            embedder=object(),  # type: ignore[arg-type]
            llm=object(),  # type: ignore[arg-type]
            ontology=Ontology(),
        )
        declared = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[Attribute(name="age", type="INTEGER", structured=True)],
                )
            ]
        )
        strategy.set_ontology(declared)
        assert strategy._ontology is declared

    def test_publishing_an_ontology_reaches_the_strategy(self, graphrag):
        """The facade republishes on every change, not only on first load.

        Measured before this: after a structured ingest the facade knew
        ``Organization.employee_count`` and the strategy's copy had no properties
        at all, so a generated query could not have aggregated over the column
        the mapping existed to declare.
        """
        declared = Ontology(
            entities=[
                Entity(
                    label="Organization",
                    properties=[Attribute(name="employee_count", type="INTEGER", structured=True)],
                )
            ]
        )
        graphrag._global_ontology = declared

        assert graphrag._global_ontology is declared
        assert graphrag._retrieval_strategy._ontology is declared

    def test_a_later_change_replaces_an_earlier_one(self, graphrag):
        first = Ontology(entities=[Entity(label="Person")])
        second = Ontology(entities=[Entity(label="Organization")])
        graphrag._global_ontology = first
        graphrag._global_ontology = second
        assert graphrag._retrieval_strategy._ontology is second


class TestPublicQuery:
    async def test_returns_rows_as_lists(self, graphrag):
        result = MagicMock()
        result.result_set = [["Acme Corp", 1200]]
        graphrag._graph_store.query_raw = AsyncMock(return_value=result)

        rows = await graphrag.query("MATCH (o:Organization) RETURN o.name, o.employee_count")
        assert rows == [["Acme Corp", 1200]]

    async def test_no_rows_is_an_empty_list_not_none(self, graphrag):
        result = MagicMock()
        result.result_set = None
        graphrag._graph_store.query_raw = AsyncMock(return_value=result)
        assert await graphrag.query("MATCH (n:Nothing) RETURN n") == []

    async def test_parameters_are_passed_through(self, graphrag):
        result = MagicMock()
        result.result_set = []
        spy = AsyncMock(return_value=result)
        graphrag._graph_store.query_raw = spy

        await graphrag.query("MATCH (o {name: $n}) RETURN o", {"n": "Acme Corp"})
        spy.assert_awaited_once_with("MATCH (o {name: $n}) RETURN o", {"n": "Acme Corp"})
