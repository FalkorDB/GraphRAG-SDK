from graphrag_sdk.ontology import Ontology
from graphrag_sdk.entity import Entity
from graphrag_sdk.relation import Relation
from graphrag_sdk.helpers import (
    validate_cypher,
    validate_cypher_entities_exist,
    validate_cypher_relations_exist,
    validate_cypher_relation_directions,
)
import unittest


import logging

logging.basicConfig(level=logging.DEBUG)


class TestValidateCypher1(unittest.TestCase):
    """
    Test a valid cypher query
    """

    cypher = """
    MATCH (f:Fighter)-[r:FOUGHT_IN]->(fight:Fight)
    RETURN f, count(fight) AS fight_count
    ORDER BY fight_count DESC
    LIMIT 1"""

    @classmethod
    def setUpClass(cls):
        cls._ontology = Ontology()

        cls._ontology.add_entity(
            Entity(
                label="Fighter",
                attributes=[],
            )
        )

        cls._ontology.add_entity(
            Entity(
                label="Fight",
                attributes=[],
            )
        )

        cls._ontology.add_relation(
            Relation(
                label="FOUGHT_IN",
                source="Fighter",
                target="Fight",
                attributes=[],
            )
        )

    def test_validate_cypher_entities_exist(self):
        errors = validate_cypher_entities_exist(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher_relations_exist(self):
        errors = validate_cypher_relations_exist(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher_relation_directions(self):
        errors = validate_cypher_relation_directions(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher(self):
        errors = validate_cypher(self.cypher, self._ontology)

        assert errors is None


class TestValidateCypher2(unittest.TestCase):
    """
    Test a cypher query with the wrong relation direction
    """

    cypher = """
    MATCH (f:Fighter)<-[r:FOUGHT_IN]-(fight:Fight)
    RETURN f"""

    @classmethod
    def setUpClass(cls):
        cls._ontology = Ontology([], [])

        cls._ontology.add_entity(
            Entity(
                label="Fighter",
                attributes=[],
            )
        )

        cls._ontology.add_entity(
            Entity(
                label="Fight",
                attributes=[],
            )
        )

        cls._ontology.add_relation(
            Relation(
                label="FOUGHT_IN",
                source="Fighter",
                target="Fight",
                attributes=[],
            )
        )

    def test_validate_cypher_entities_exist(self):
        errors = validate_cypher_entities_exist(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher_relations_exist(self):
        errors = validate_cypher_relations_exist(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher_relation_directions(self):
        errors = validate_cypher_relation_directions(self.cypher, self._ontology)

        assert len(errors) == 1

    def test_validate_cypher(self):
        errors = validate_cypher(self.cypher, self._ontology)

        assert errors is not None


class TestValidateCypher3(unittest.TestCase):
    """
    Test a cypher query with multiple right edge directions
    """

    cypher = """
    MATCH (a:Airline)-[:ACCEPTS]->(p:Pet), (r:Route)-[:ALLOWS]->(sd:Service_Dog)
    RETURN a, p, r, sd
    """

    @classmethod
    def setUpClass(cls):
        cls._ontology = Ontology([], [])

        cls._ontology.add_entity(
            Entity(
                label="Airline",
                attributes=[],
            )
        )

        cls._ontology.add_entity(
            Entity(
                label="Pet",
                attributes=[],
            )
        )

        cls._ontology.add_entity(
            Entity(
                label="Route",
                attributes=[],
            )
        )

        cls._ontology.add_entity(
            Entity(
                label="Service_Dog",
                attributes=[],
            )
        )

        cls._ontology.add_relation(
            Relation(
                label="ACCEPTS",
                source="Airline",
                target="Pet",
                attributes=[],
            )
        )

        cls._ontology.add_relation(
            Relation(
                label="ALLOWS",
                source="Route",
                target="Service_Dog",
                attributes=[],
            )
        )

    def test_validate_cypher_nodes_exist(self):
        errors = validate_cypher_entities_exist(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher_edges_exist(self):
        errors = validate_cypher_relations_exist(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher_edge_directions(self):
        errors = validate_cypher_relation_directions(self.cypher, self._ontology)

        assert len(errors) == 0

    def test_validate_cypher(self):
        errors = validate_cypher(self.cypher, self._ontology)

        assert errors is None or len(errors) == 0


if __name__ == "__main__":
    unittest.main()
