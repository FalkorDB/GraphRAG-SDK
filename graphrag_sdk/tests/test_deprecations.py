"""Verify the v1.2 ontology-vocabulary deprecation aliases.

Old code that imports ``GraphSchema``, ``EntityType``, ``RelationType``,
``PropertyType`` or ``SchemaModificationNotAllowedError`` must keep working
— the symbols resolve to the new names and a :py:class:`DeprecationWarning`
fires on each access. Same for the legacy ``schema=`` kwarg and ``rag.schema``
attribute on :py:class:`GraphRAG`.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation


class _StubLLM:
    async def ainvoke(self, prompt, **_):
        return None

    async def abatch_invoke(self, prompts, **_):
        return []


class _StubEmbedder:
    async def aembed_query(self, q, **_):
        return [0.0] * 8

    async def aembed_documents(self, ds, **_):
        return [[0.0] * 8 for _ in ds]

    def embed_query(self, q, **_):
        return [0.0] * 8

    def embed_documents(self, ds, **_):
        return [[0.0] * 8 for _ in ds]


# ── module-level class aliases ───────────────────────────────────


@pytest.mark.parametrize(
    "old,new",
    [
        ("GraphSchema", "Ontology"),
        ("EntityType", "Entity"),
        ("RelationType", "Relation"),
        ("PropertyType", "Attribute"),
    ],
)
class TestModuleLevelClassAliases:
    def test_top_level_import_warns(self, old, new):
        import graphrag_sdk

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = getattr(graphrag_sdk, old)
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deps, f"expected DeprecationWarning for graphrag_sdk.{old}"
        assert old in str(deps[0].message)
        assert new in str(deps[0].message)

    def test_top_level_resolves_to_new_class(self, old, new):
        import graphrag_sdk

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old_cls = getattr(graphrag_sdk, old)
            new_cls = getattr(graphrag_sdk, new)
        assert old_cls is new_cls

    def test_models_module_import_warns(self, old, new):
        from graphrag_sdk.core import models as _models

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = getattr(_models, old)
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deps, f"expected DeprecationWarning for core.models.{old}"


def test_instantiating_via_legacy_alias_returns_new_class():
    """Old code: GraphSchema(entities=[EntityType(...)]) keeps working."""
    import graphrag_sdk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_schema = graphrag_sdk.GraphSchema(
            entities=[
                graphrag_sdk.EntityType(
                    label="Person",
                    properties=[
                        graphrag_sdk.PropertyType(name="age", type="INTEGER"),
                    ],
                )
            ]
        )
    assert isinstance(legacy_schema, Ontology)
    assert len(legacy_schema.entities) == 1
    assert isinstance(legacy_schema.entities[0], Entity)
    assert legacy_schema.entities[0].label == "Person"
    assert isinstance(legacy_schema.entities[0].properties[0], Attribute)


# ── exception alias ──────────────────────────────────────────────


class TestExceptionAlias:
    def test_legacy_import_from_ontology_store_warns(self):
        from graphrag_sdk.storage import ontology_store as _store

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls = _store.SchemaModificationNotAllowedError
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deps
        assert cls is _store.OntologyModificationNotAllowedError

    def test_legacy_top_level_import_warns(self):
        import graphrag_sdk
        from graphrag_sdk.storage.ontology_store import (
            OntologyModificationNotAllowedError,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls = graphrag_sdk.SchemaModificationNotAllowedError
        deps = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deps
        assert cls is OntologyModificationNotAllowedError


# ── GraphRAG schema= kwarg and rag.schema property ───────────────


def _make_graphrag(**kwargs):
    from graphrag_sdk import GraphRAG

    return GraphRAG(
        connection=ConnectionConfig(graph_name=f"test_dep_{id(kwargs)}"),
        llm=_StubLLM(),
        embedder=_StubEmbedder(),
        **kwargs,
    )


class TestGraphRAGKwargAlias:
    def test_schema_kwarg_emits_deprecation_warning(self):
        ontology = Ontology(entities=[Entity(label="Person")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rag = _make_graphrag(schema=ontology)
        deps = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "schema=" in str(x.message)
        ]
        assert deps, "expected DeprecationWarning when passing schema= kwarg"
        assert rag.ontology is ontology

    def test_both_ontology_and_schema_raises(self):
        with pytest.raises(TypeError, match="both"):
            _make_graphrag(ontology=Ontology(), schema=Ontology())

    def test_only_ontology_does_not_warn(self):
        ontology = Ontology(entities=[Entity(label="Person")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rag = _make_graphrag(ontology=ontology)
        deps = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "schema=" in str(x.message)
        ]
        assert not deps
        assert rag.ontology is ontology


class TestGraphRAGAttributeAlias:
    def test_reading_rag_schema_warns_and_returns_ontology(self):
        ontology = Ontology(entities=[Entity(label="Person")])
        rag = _make_graphrag(ontology=ontology)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = rag.schema
        deps = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "GraphRAG.schema" in str(x.message)
        ]
        assert deps
        assert out is rag.ontology

    def test_writing_rag_schema_warns_and_updates_ontology(self):
        rag = _make_graphrag(ontology=Ontology())
        new_ontology = Ontology(entities=[Entity(label="Company")])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rag.schema = new_ontology
        deps = [
            x
            for x in w
            if issubclass(x.category, DeprecationWarning)
            and "GraphRAG.schema" in str(x.message)
        ]
        assert deps
        assert rag.ontology is new_ontology
