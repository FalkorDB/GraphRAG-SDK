"""Unit tests for ontology evolution — Groups 1 (pure ontology) and 2
(mechanical data migration).

OntologyStore evolution primitives are exercised against the existing
``_FakeGraph`` from ``test_ontology_store.py``. GraphRAG orchestration
methods are exercised with mocked ``OntologyStore`` and ``GraphStore``
so we can assert on call ordering and input validation without booting
FalkorDB.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.connection import ConnectionConfig, FalkorDBConnection
from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation
from graphrag_sdk.storage.ontology_store import OntologyStore

from .test_ontology_store import _FakeGraph

# ── Shared fixtures ──────────────────────────────────────────────


@pytest.fixture
def store_factory():
    def _make() -> tuple[OntologyStore, _FakeGraph]:
        fake = _FakeGraph()
        conn = MagicMock()
        conn._ensure_client = MagicMock()
        conn._driver = SimpleNamespace(select_graph=MagicMock(return_value=fake))
        return OntologyStore(conn, "kg"), fake

    return _make


@pytest.fixture
def evolving_ontology() -> Ontology:
    """Ontology used by GraphRAG orchestration tests — must contain at
    least one Entity (with attribute), one Relation (with pattern), so
    every method has something to operate on."""
    return Ontology(
        entities=[
            Entity(
                label="Person",
                description="A human",
                properties=[Attribute(name="age", type="INTEGER")],
            ),
            Entity(label="Company"),
        ],
        relations=[
            Relation(
                label="WORKS_AT",
                patterns=[("Person", "Company")],
            ),
        ],
    )


@pytest.fixture
def graphrag_evolving(embedder, llm, evolving_ontology, mock_graph_store):
    """A GraphRAG instance with mocked stores. ``_global_ontology`` is
    pre-set so we can call evolution methods without firing the lazy
    ontology load (which would hit the mocked connection).
    """
    conn = MagicMock(spec=FalkorDBConnection)
    conn.config = ConnectionConfig()
    rag = GraphRAG(connection=conn, llm=llm, embedder=embedder, embedding_dimension=8)
    rag._graph_store = mock_graph_store
    rag._ontology_store = MagicMock(spec=OntologyStore)
    rag._ontology_store.load = AsyncMock(return_value=evolving_ontology)
    rag._ontology_store.register = AsyncMock(return_value=evolving_ontology)
    rag._ontology_store.add_entity_property = AsyncMock()
    rag._ontology_store.add_relation_property = AsyncMock()
    rag._ontology_store.add_relation_pattern_node = AsyncMock()
    rag._ontology_store.rename_entity_label = AsyncMock()
    rag._ontology_store.rename_relation_label = AsyncMock()
    rag._ontology_store.rename_property_label = AsyncMock()
    rag._ontology_store.set_description = AsyncMock()
    rag._ontology_store.drop_entity_property = AsyncMock()
    rag._ontology_store.drop_relation_property = AsyncMock()
    rag._ontology_store.drop_entity_label = AsyncMock()
    rag._ontology_store.drop_relation_label = AsyncMock()
    rag._ontology_store.drop_relation_pattern_node = AsyncMock()
    # Pretend the lazy init has already happened so methods don't try to
    # re-register the (mocked-away) initial ontology.
    rag._ontology_initialized = True
    rag._global_ontology = evolving_ontology
    rag.ontology = evolving_ontology
    # GraphStore add primitives needed by orchestrators
    rag._graph_store.rename_label = AsyncMock(return_value=3)
    rag._graph_store.rename_node_property = AsyncMock(return_value=2)
    rag._graph_store.rename_relation_type = AsyncMock(return_value=4)
    rag._graph_store.drop_node_property = AsyncMock(return_value=2)
    rag._graph_store.delete_nodes_by_label = AsyncMock(return_value=5)
    rag._graph_store.delete_relations_by_type = AsyncMock(return_value=7)
    rag._graph_store.delete_relations_by_pattern = AsyncMock(return_value=3)
    return rag


# ── OntologyStore primitives ─────────────────────────────────────


class TestAddEntityProperty:
    @pytest.mark.asyncio
    async def test_merges_property_with_owner_scope(self, store_factory):
        store, fake = store_factory()
        await store.add_entity_property(
            "Person", Attribute(name="email", type="STRING", description="email addr")
        )
        prop_merges = [
            c
            for c in fake.calls
            if "MATCH (e:Entity {label: $owner})" in c[0]
            and "MERGE (e)-[:HAS_PROPERTY]->(p:Property {label: $name})" in c[0]
        ]
        assert len(prop_merges) == 1
        params = prop_merges[0][1]
        assert params["owner"] == "Person"
        assert params["name"] == "email"
        assert params["type"] == "STRING"


class TestAddRelationProperty:
    @pytest.mark.asyncio
    async def test_merges_on_every_relation_node_with_label(self, store_factory):
        store, fake = store_factory()
        await store.add_relation_property("WORKS_AT", Attribute(name="since", type="DATE"))
        merges = [
            c
            for c in fake.calls
            if "MATCH (r:Relation {label: $rel_label})" in c[0]
            and "MERGE (r)-[:HAS_PROPERTY]->(p:Property {label: $name})" in c[0]
        ]
        assert len(merges) == 1
        params = merges[0][1]
        assert params == {
            "rel_label": "WORKS_AT",
            "name": "since",
            "type": "DATE",
            "description": None,
        }


class TestAddRelationPatternNode:
    @pytest.mark.asyncio
    async def test_matches_entities_does_not_merge_them(self, store_factory):
        """Endpoints must MATCH, not MERGE. The public facade validates the
        endpoint labels already; if the OntologyStore primitive MERGEd, a
        stray direct call would create phantom :Entity nodes — exactly the
        drift the alignment invariant exists to prevent.
        """
        store, fake = store_factory()
        await store.add_relation_pattern_node("WORKS_AT", "Person", "Startup")
        # The Cypher MATCHes the endpoint entities (does NOT MERGE them) and
        # only MERGEs the new pattern node.
        match_calls = [c for c in fake.calls if "MATCH (s:Entity {label: $src})" in c[0]]
        assert len(match_calls) == 1
        # Must NOT MERGE the endpoints.
        for cypher, _ in fake.calls:
            assert "MERGE (s:Entity {label: $src})" not in cypher
            assert "MERGE (t:Entity {label: $tgt})" not in cypher
        copy_calls = [
            c
            for c in fake.calls
            if "MATCH (existing:Relation {label: $rel_label})" in c[0]
            and "MERGE (new)-[:HAS_PROPERTY]" in c[0]
        ]
        assert len(copy_calls) == 1


class TestRenameLabel:
    @pytest.mark.asyncio
    async def test_rename_entity_label(self, store_factory):
        store, fake = store_factory()
        await store.rename_entity_label("Person", "Human")
        ent_renames = [c for c in fake.calls if "SET e.label = $new" in c[0]]
        assert len(ent_renames) == 1
        assert ent_renames[0][1] == {"old": "Person", "new": "Human"}

    @pytest.mark.asyncio
    async def test_rename_relation_label(self, store_factory):
        store, fake = store_factory()
        await store.rename_relation_label("WORKS_AT", "EMPLOYED_AT")
        rel_renames = [c for c in fake.calls if "SET r.label = $new" in c[0]]
        assert len(rel_renames) == 1


class TestRenameProperty:
    @pytest.mark.asyncio
    async def test_entity_property_rename_scoped_to_owner(self, store_factory):
        store, fake = store_factory()
        await store.rename_property_label("entity", "Person", "age", "years_old")
        renames = [c for c in fake.calls if "SET p.label = $new_name" in c[0]]
        assert len(renames) == 1
        assert "MATCH (o:Entity {label: $owner})" in renames[0][0]

    @pytest.mark.asyncio
    async def test_relation_property_rename_uses_relation_match(self, store_factory):
        store, fake = store_factory()
        await store.rename_property_label("relation", "WORKS_AT", "since", "start_date")
        renames = [c for c in fake.calls if "SET p.label = $new_name" in c[0]]
        assert len(renames) == 1
        assert "MATCH (o:Relation {label: $owner})" in renames[0][0]


class TestSetDescription:
    @pytest.mark.asyncio
    async def test_entity_description(self, store_factory):
        store, fake = store_factory()
        await store.set_description("entity", "Person", "A human being")
        updates = [c for c in fake.calls if "SET e.description = $description" in c[0]]
        assert len(updates) == 1
        assert updates[0][1] == {"label": "Person", "description": "A human being"}

    @pytest.mark.asyncio
    async def test_property_description_requires_owner(self, store_factory):
        store, _ = store_factory()
        with pytest.raises(ValueError, match="owner_label"):
            await store.set_description("entity_property", "age", "Years")

    @pytest.mark.asyncio
    async def test_rejects_unknown_kind(self, store_factory):
        store, _ = store_factory()
        with pytest.raises(ValueError, match="Unknown description kind"):
            await store.set_description("widget", "Person", "x")  # type: ignore[arg-type]


class TestDropProperty:
    @pytest.mark.asyncio
    async def test_drop_entity_property(self, store_factory):
        store, fake = store_factory()
        await store.drop_entity_property("Person", "age")
        deletes = [c for c in fake.calls if "DETACH DELETE p" in c[0] and "Entity" in c[0]]
        assert len(deletes) == 1


class TestDropEntityLabel:
    @pytest.mark.asyncio
    async def test_cascades_relations_referencing_the_entity(self, store_factory):
        store, fake = store_factory()
        await store.drop_entity_label("Company")
        # Three Cypher writes: drop entity props, drop relation pattern nodes
        # that reference the entity, drop the entity node itself.
        assert any(
            "DETACH DELETE p" in c[0] and "(e:Entity {label: $label})-[:HAS_PROPERTY]" in c[0]
            for c in fake.calls
        )
        assert any("[:SOURCE|TARGET]->(e:Entity {label: $label})" in c[0] for c in fake.calls)
        assert any("MATCH (e:Entity {label: $label}) DETACH DELETE e" in c[0] for c in fake.calls)


class TestAddPropertyRefusesRetype:
    """``add_entity_property`` / ``add_relation_property`` must NOT silently
    retype an existing property — without this guard a stray
    ``add_attribute('Person', Attribute(name='age', type='STRING'))`` on a
    Person.age INTEGER would mutate the schema in place. To change a
    declared type, callers must go through ``GraphRAG.drop_attribute`` +
    ``GraphRAG.add_attribute`` with the new type."""

    @pytest.mark.asyncio
    async def test_add_entity_property_raises_on_type_conflict(self, store_factory):
        from graphrag_sdk.storage.ontology_store import OntologyContradictionError

        store, fake = store_factory()

        # Patch the type-check probe to report an existing INTEGER.
        async def _scripted_query(cypher, params=None):
            fake.calls.append((cypher, params))
            from types import SimpleNamespace

            if "RETURN p.type AS type LIMIT 1" in cypher:
                return SimpleNamespace(result_set=[["INTEGER"]])
            return SimpleNamespace(result_set=[])

        fake.query = _scripted_query  # type: ignore[method-assign]
        with pytest.raises(OntologyContradictionError, match="already registered"):
            await store.add_entity_property("Person", Attribute(name="age", type="STRING"))
        # No SET p.type write — the contradiction check fires before the upsert.
        assert not any(
            "MERGE (e)-[:HAS_PROPERTY]" in c[0] and "SET p.type" in c[0] for c in fake.calls
        )


# ── GraphRAG orchestration: Group 1 ──────────────────────────────


class TestSetDescriptionOrchestration:
    @pytest.mark.asyncio
    async def test_set_entity_description_rejects_unknown_label(self, graphrag_evolving):
        with pytest.raises(ValueError, match="Unknown entity"):
            await graphrag_evolving.set_entity_description("Alien", "x")

    @pytest.mark.asyncio
    async def test_set_entity_description_calls_ontology_store(self, graphrag_evolving):
        await graphrag_evolving.set_entity_description("Person", "A human")
        graphrag_evolving._ontology_store.set_description.assert_awaited_once_with(
            "entity", "Person", "A human"
        )

    @pytest.mark.asyncio
    async def test_set_attribute_description_passes_owner_kind(self, graphrag_evolving):
        await graphrag_evolving.set_attribute_description("Person", "age", "Years")
        graphrag_evolving._ontology_store.set_description.assert_awaited_once_with(
            "entity_property", "age", "Years", owner_label="Person"
        )


class TestAddDeclarations:
    @pytest.mark.asyncio
    async def test_add_entity_rejects_duplicate(self, graphrag_evolving):
        with pytest.raises(ValueError, match="already exists"):
            await graphrag_evolving.add_entity(Entity(label="Person"))

    @pytest.mark.asyncio
    async def test_add_entity_registers_new_label(self, graphrag_evolving):
        await graphrag_evolving.add_entity(Entity(label="University"))
        graphrag_evolving._ontology_store.register.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_attribute_requires_known_label(self, graphrag_evolving):
        with pytest.raises(ValueError, match="Unknown ontology label"):
            await graphrag_evolving.add_attribute("Alien", Attribute(name="x"))

    @pytest.mark.asyncio
    async def test_add_attribute_on_relation_raises_not_implemented(self, graphrag_evolving):
        """Under the strict alignment design, attribute evolution on relation
        owners would require iterating edges. Relation-attribute mutation
        raises NotImplementedError until that's implemented; the workaround
        is documented in the error message."""
        with pytest.raises(NotImplementedError, match="relation owners"):
            await graphrag_evolving.add_attribute("WORKS_AT", Attribute(name="since"))
        # The data graph must not have been touched either.
        graphrag_evolving._ontology_store.add_relation_property.assert_not_awaited()
        graphrag_evolving._ontology_store.add_entity_property.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_add_attribute_rejects_already_declared(self, graphrag_evolving):
        """add_attribute is an atomic 'declare + backfill'. Re-declaring an
        existing attribute would either be a no-op (wasteful) or a type
        change (drift-prone). For type changes the user must drop_attribute
        first — the LLM then re-derives values from chunks."""
        with pytest.raises(ValueError, match="already declared"):
            await graphrag_evolving.add_attribute("Person", Attribute(name="age", type="STRING"))

    @pytest.mark.asyncio
    async def test_add_relation_pattern_requires_known_endpoints(self, graphrag_evolving):
        with pytest.raises(ValueError, match="source entity"):
            await graphrag_evolving.add_relation_pattern("WORKS_AT", "Alien", "Company")


# ── GraphRAG orchestration: Group 2 ──────────────────────────────


class TestRenameEntity:
    @pytest.mark.asyncio
    async def test_data_migration_runs_before_ontology_update(self, graphrag_evolving):
        await graphrag_evolving.rename_entity("Person", "Human")
        graphrag_evolving._graph_store.rename_label.assert_awaited_once_with("Person", "Human")
        graphrag_evolving._ontology_store.rename_entity_label.assert_awaited_once_with(
            "Person", "Human"
        )

    @pytest.mark.asyncio
    async def test_rejects_existing_target(self, graphrag_evolving):
        with pytest.raises(ValueError, match="merging entity types"):
            await graphrag_evolving.rename_entity("Person", "Company")

    @pytest.mark.asyncio
    async def test_no_op_when_old_equals_new(self, graphrag_evolving):
        await graphrag_evolving.rename_entity("Person", "Person")
        graphrag_evolving._graph_store.rename_label.assert_not_awaited()


class TestRenameAttributeValidation:
    """``rename_attribute`` must refuse silent no-ops (typoed old_name) and
    silent overwrites (collision on new_name) — without this guard a
    misspelled rename does nothing or stamps over existing values."""

    @pytest.mark.asyncio
    async def test_rejects_missing_old_name(self, graphrag_evolving):
        with pytest.raises(ValueError, match="Unknown attribute"):
            await graphrag_evolving.rename_attribute("Person", "nonexistent", "new")
        graphrag_evolving._graph_store.rename_node_property.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_existing_new_name(self, graphrag_evolving, evolving_ontology):
        # Add a second attribute so we can test collision detection.
        person = next(e for e in evolving_ontology.entities if e.label == "Person")
        person.properties.append(Attribute(name="email", type="STRING"))
        with pytest.raises(ValueError, match="already exists"):
            await graphrag_evolving.rename_attribute("Person", "age", "email")
        graphrag_evolving._graph_store.rename_node_property.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_relation_owner_raises_not_implemented(self, graphrag_evolving):
        """Under the strict alignment design, rename on a relation owner
        used to half-apply: ontology graph renamed, edges kept their old
        property key — silent drift. Must raise NotImplementedError,
        matching add_attribute / drop_attribute on relations."""
        with pytest.raises(NotImplementedError, match="relation owners"):
            await graphrag_evolving.rename_attribute("WORKS_AT", "since", "started")
        graphrag_evolving._graph_store.rename_node_property.assert_not_awaited()
        graphrag_evolving._ontology_store.rename_property_label.assert_not_awaited()


class TestRefreshGlobalOntologySyncsSelf:
    """``_refresh_global_ontology`` must update ``self.ontology`` in lockstep
    so a later ``ingest()`` (which reads ``self.ontology.entities`` in
    ``_default_extractor``) sees the post-mutation label set."""

    @pytest.mark.asyncio
    async def test_self_ontology_updated(self, graphrag_evolving):
        new_loaded = Ontology(entities=[Entity(label="Updated")])
        graphrag_evolving._ontology_store.load = AsyncMock(return_value=new_loaded)
        await graphrag_evolving._refresh_global_ontology()
        assert graphrag_evolving._global_ontology is new_loaded
        assert graphrag_evolving.ontology is new_loaded


class TestRenameRelation:
    @pytest.mark.asyncio
    async def test_orchestrates_recreate_then_ontology(self, graphrag_evolving):
        await graphrag_evolving.rename_relation("WORKS_AT", "EMPLOYED_AT")
        graphrag_evolving._graph_store.rename_relation_type.assert_awaited_once_with(
            "WORKS_AT", "EMPLOYED_AT"
        )
        graphrag_evolving._ontology_store.rename_relation_label.assert_awaited_once_with(
            "WORKS_AT", "EMPLOYED_AT"
        )

    @pytest.mark.asyncio
    async def test_rejects_unknown(self, graphrag_evolving):
        with pytest.raises(ValueError, match="Unknown relation"):
            await graphrag_evolving.rename_relation("UNKNOWN", "NEW")


class TestDropAttribute:
    @pytest.mark.asyncio
    async def test_entity_owner_drops_data_and_ontology(self, graphrag_evolving):
        await graphrag_evolving.drop_attribute("Person", "age")
        graphrag_evolving._graph_store.drop_node_property.assert_awaited_once_with("Person", "age")
        graphrag_evolving._ontology_store.drop_entity_property.assert_awaited_once_with(
            "Person", "age"
        )

    @pytest.mark.asyncio
    async def test_relation_owner_raises_not_implemented(self, graphrag_evolving):
        """Drop on a relation owner would leave edge properties orphaned
        (no edge-property removal primitive yet). Raises under the strict
        alignment invariant to prevent silent drift."""
        with pytest.raises(NotImplementedError, match="relation owners"):
            await graphrag_evolving.drop_attribute("WORKS_AT", "since")
        graphrag_evolving._graph_store.drop_node_property.assert_not_awaited()
        graphrag_evolving._ontology_store.drop_relation_property.assert_not_awaited()


class TestDropEntity:
    @pytest.mark.asyncio
    async def test_deletes_data_then_ontology(self, graphrag_evolving):
        await graphrag_evolving.drop_entity("Company")
        graphrag_evolving._graph_store.delete_nodes_by_label.assert_awaited_once_with("Company")
        graphrag_evolving._ontology_store.drop_entity_label.assert_awaited_once_with("Company")


class TestDropRelation:
    @pytest.mark.asyncio
    async def test_deletes_edges_then_ontology(self, graphrag_evolving):
        await graphrag_evolving.drop_relation("WORKS_AT")
        graphrag_evolving._graph_store.delete_relations_by_type.assert_awaited_once_with("WORKS_AT")
        graphrag_evolving._ontology_store.drop_relation_label.assert_awaited_once_with("WORKS_AT")


class TestDropRelationPattern:
    @pytest.mark.asyncio
    async def test_targets_only_the_named_pattern(self, graphrag_evolving):
        await graphrag_evolving.drop_relation_pattern("WORKS_AT", "Person", "Company")
        graphrag_evolving._graph_store.delete_relations_by_pattern.assert_awaited_once_with(
            "WORKS_AT", "Person", "Company"
        )
        graphrag_evolving._ontology_store.drop_relation_pattern_node.assert_awaited_once_with(
            "WORKS_AT", "Person", "Company"
        )


class TestRetypeRemoved:
    """Under the strict alignment design, ``retype_attribute`` is gone.
    Users compose ``drop_attribute`` + ``add_attribute`` with the new type
    — the LLM re-derives values from chunks. Re-deriving is the only
    honest move when the ontology's source of truth is the corpus."""

    def test_retype_attribute_does_not_exist(self, graphrag_evolving):
        assert not hasattr(graphrag_evolving, "retype_attribute")
