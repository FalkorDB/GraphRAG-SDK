"""Asking the model what a table is, and holding it to the data.

A table nobody declared a mapping for gets one proposed: the model is shown the
measured columns, the first rows and the ontology, and answers with a
``MappingProposal``. Every claim it makes about the *data* is then checked
against the file — the columns exist, the key is unique, the types parse — and a
claim that does not hold goes back to the model as the exact reason, or is
corrected from the measurement when the measurement is the authority.

No database. The model is a ``MockLLM`` with scripted answers.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import Entity, Ontology, Relation
from graphrag_sdk.core.tables import Column, TableMapping
from graphrag_sdk.ingestion.loaders.record_loader import CsvRecordLoader
from graphrag_sdk.ingestion.mapping import MappingError
from graphrag_sdk.ingestion.mapping_proposal import propose_mapping

from .conftest import MockLLM

EMPLOYEES = (
    "employee_id,full_name,age,org_id,joined\n"
    "E-1,Maya Ellison,34,ORG-1,2019-04-01\n"
    "E-2,Tomas Reyes,47,ORG-2,2015-11-01\n"
    "E-3,Lene Holm,29,ORG-1,2021-01-15\n"
)

GOOD = {
    "label": "Person",
    "name": "full_name",
    "key": "employee_id",
    "properties": [
        {"column": "age", "type": "INTEGER"},
        {"column": "joined", "property": "joined_on", "type": "DATE"},
    ],
    "links": [{"column": "org_id", "type": "WORKS_AT", "to": "Organization"}],
    "reasoning": "Each row is one employee, a person, who works at an organization.",
}

ONTOLOGY = Ontology(
    entities=[Entity(label="Person"), Entity(label="Organization")],
    relations=[Relation(label="WORKS_AT", patterns=[("Person", "Organization")])],
    tables=[
        TableMapping(
            source="orgs.csv",
            label="Organization",
            key="org_id",
            name="org_name",
            properties={"hq_country": Column("hq_country", "STRING")},
        )
    ],
)


async def _batch(text: str = EMPLOYEES, name: str = "employees.csv"):
    path = pathlib.Path(tempfile.mkdtemp()) / name
    path.write_text(text)
    return await CsvRecordLoader().load_records(str(path), Context()), str(path)


def _feedback(llm: MockLLM) -> str:
    """What the model was told after its last rejected answer."""
    return next(m.content for m in reversed(llm.last_messages) if m.role == "user")


def _llm(*answers: dict | str) -> MockLLM:
    return MockLLM([a if isinstance(a, str) else json.dumps(a) for a in answers], strict=True)


async def _propose(llm: MockLLM, text: str = EMPLOYEES, ontology: Ontology = ONTOLOGY):
    batch, path = await _batch(text)
    return await propose_mapping(batch, path, llm=llm, ontology=ontology)


class TestAGoodAnswerBecomesTheMapping:
    async def test_the_proposal_is_the_declaration_the_user_would_have_written(self):
        mapping, notes = await _propose(_llm(GOOD))

        assert mapping.label == "Person"
        assert mapping.name == "full_name"
        assert mapping.key == "employee_id"
        assert {n: c.type for n, c in mapping.typed_properties.items()} == {
            "age": "INTEGER",
            "joined_on": "DATE",
        }
        assert [(link.type, link.to, link.by) for link in mapping.links] == [
            ("WORKS_AT", "Organization", "org_id")
        ]
        assert mapping.derived is True, "finalize() must be able to tell it was proposed"
        assert mapping.standalone is False
        assert notes[0].startswith("label 'Person'")

    async def test_it_is_stored_under_the_tables_name_not_its_path(self):
        """A user later declares ``TableMapping(source="employees.csv")``; that
        has to land on this stored mapping, so this one is keyed the same way."""
        mapping, _ = await _propose(_llm(GOOD))
        assert mapping.source == "employees.csv"
        assert mapping.signature == "employees"

    async def test_the_model_is_shown_what_the_graph_already_holds(self):
        llm = _llm(GOOD)
        await _propose(llm)
        prompt = llm.last_messages[1].content
        assert "Person" in prompt and "Organization" in prompt
        assert "WORKS_AT" in prompt
        assert "orgs.csv -> Organization, key column 'org_id'" in prompt
        assert "employee_id: STRING; unique, filled on every row" in prompt
        assert "Maya Ellison" in prompt, "the first rows are shown verbatim"

    async def test_it_is_one_call_for_the_whole_table(self):
        llm = _llm(GOOD)
        await _propose(llm)
        assert llm._call_index == 1


class TestTheDataHasTheLastWord:
    async def test_a_column_that_is_not_in_the_header_is_sent_back(self):
        wrong = dict(GOOD, name="fullname")
        llm = _llm(wrong, GOOD)
        mapping, _ = await _propose(llm)

        assert mapping.name == "full_name"
        assert llm._call_index == 2
        assert "'fullname'" in _feedback(llm) and "not in the header" in _feedback(llm)

    async def test_a_key_that_is_not_unique_is_sent_back(self):
        wrong = dict(GOOD, key="org_id")
        llm = _llm(wrong, GOOD)
        mapping, _ = await _propose(llm)

        assert mapping.key == "employee_id"
        assert "'org_id' is not unique" in _feedback(llm)

    async def test_a_repeating_name_needs_a_key(self):
        rows = "employee_id,full_name\nE-1,John Smith\nE-2,John Smith\n"
        wrong = {"label": "Person", "name": "full_name", "properties": []}
        right = dict(wrong, key="employee_id")
        llm = _llm(wrong, right)
        mapping, _ = await _propose(llm, text=rows)

        assert mapping.key == "employee_id"
        assert "repeats" in _feedback(llm)

    async def test_a_link_to_a_label_nothing_knows_is_sent_back(self):
        wrong = dict(GOOD, links=[{"column": "org_id", "type": "WORKS_AT", "to": "Company"}])
        llm = _llm(wrong, GOOD)
        mapping, _ = await _propose(llm)

        assert mapping.links[0].to == "Organization"
        assert "'Company'" in _feedback(llm)

    async def test_a_type_the_file_does_not_hold_is_widened_not_refused(self):
        """Whether every cell parses as INTEGER is a fact about the file; the
        model does not get a vote, and is not made to pay a retry for it."""
        rows = "employee_id,full_name,age\nE-1,Maya Ellison,34\nE-2,Tomas Reyes,N/A\n"
        answer = {
            "label": "Person",
            "name": "full_name",
            "key": "employee_id",
            "properties": [{"column": "age", "type": "INTEGER"}],
        }
        llm = _llm(answer)
        mapping, notes = await _propose(llm, text=rows)

        assert mapping.typed_properties["age"].type == "STRING"
        assert llm._call_index == 1
        assert any("age kept as STRING, not INTEGER" in note for note in notes)

    async def test_a_wider_type_than_measured_is_the_models_call(self):
        """Zip codes and ids parse as integers; the model may know better."""
        answer = dict(GOOD, properties=[{"column": "age", "type": "STRING"}])
        mapping, _ = await _propose(_llm(answer))
        assert mapping.typed_properties["age"].type == "STRING"

    async def test_a_column_the_model_forgot_is_kept(self):
        answer = dict(GOOD, properties=[{"column": "age", "type": "INTEGER"}])
        mapping, notes = await _propose(_llm(answer))

        assert mapping.typed_properties["joined"].type == "DATE"
        assert any("joined was not mentioned; kept as DATE" in note for note in notes)

    async def test_a_column_cannot_be_both_the_name_and_a_property(self):
        wrong = dict(GOOD, properties=GOOD["properties"] + [{"column": "full_name"}])
        llm = _llm(wrong, GOOD)
        mapping, _ = await _propose(llm)
        assert "full_name" not in mapping.typed_properties
        assert "also the name" in _feedback(llm)

    async def test_an_awkward_property_name_is_made_usable(self):
        answer = dict(GOOD, properties=[{"column": "age", "property": "Age (years)"}])
        mapping, _ = await _propose(_llm(answer))
        assert "col_age_years" in mapping.typed_properties


class TestWhenTheModelCannotAnswer:
    async def test_running_out_of_retries_raises_so_the_caller_can_fall_back(self):
        llm = MockLLM(['{"nodes": [], "relationships": []}'])
        with pytest.raises(MappingError, match="did not produce an acceptable mapping"):
            await _propose(llm)
        assert llm._call_index == 3, "one call plus max_retries=2"

    async def test_a_proposal_without_name_or_key_is_sent_back(self):
        wrong = {"label": "Person", "properties": []}
        llm = _llm(wrong, GOOD)
        await _propose(llm)
        assert "neither name nor key" in _feedback(llm)

    async def test_fences_around_the_json_are_tolerated(self):
        mapping, _ = await _propose(_llm("```json\n" + json.dumps(GOOD) + "\n```"))
        assert mapping.label == "Person"
