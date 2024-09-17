import json
from falkordb import Graph
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.models import GenerativeModel
import graphrag_sdk
import logging
from .relation import Relation
from .entity import Entity
from typing import Optional


logger = logging.getLogger(__name__)


class Ontology(object):
    """
    Represents an ontology, which is a collection of entities and relations.

    Attributes:
        entities (list[Entity]): The list of entities in the ontology.
        relations (list[Relation]): The list of relations in the ontology.
    """

    def __init__(self, entities: list[Entity] = None, relations: list[Relation] = None):
        """
        Initialize the Ontology class.

        Args:
            entities (list[Entity], optional): List of Entity objects. Defaults to None.
            relations (list[Relation], optional): List of Relation objects. Defaults to None.
        """
        self.entities = entities or []
        self.relations = relations or []

    @staticmethod
    def from_sources(
        sources: list[AbstractSource],
        model: GenerativeModel,
        boundaries: Optional[str] = None,
    ) -> "Ontology":
        """
        Create an Ontology object from a list of sources.

        Args:
            sources (list[AbstractSource]): A list of AbstractSource objects representing the sources.
            boundaries (Optinal[str]): The boundaries for the ontology.
            model (GenerativeModel): The generative model to use.

        Returns:
            The created Ontology object.
        """
        step = graphrag_sdk.CreateOntologyStep(
            sources=sources,
            ontology=Ontology(),
            model=model,
        )

        return step.run(boundaries=boundaries)

    @staticmethod
    def from_json(txt: dict | str):
        """
        Creates an Ontology object from a JSON representation.

        Args:
            txt (dict | str): The JSON representation of the ontology. It can be either a dictionary or a string.

        Returns:
            The Ontology object created from the JSON representation.

        Raises:
            ValueError: If the provided JSON representation is invalid.
        """
        txt = txt if isinstance(txt, dict) else json.loads(txt)
        return Ontology(
            [Entity.from_json(entity) for entity in txt["entities"]],
            [Relation.from_json(relation) for relation in txt["relations"]],
        )

    @staticmethod
    def from_graph(graph: Graph):
        """
        Creates an Ontology object from a given graph.

        Args:
            graph (Graph): The graph object representing the ontology.

        Returns:
            The Ontology object created from the graph.
        """
        ontology = Ontology()

        entities = graph.query("MATCH (n) RETURN n").result_set
        for entity in entities:
            ontology.add_entity(Entity.from_graph(entity[0]))

        for relation in graph.query("MATCH ()-[r]->() RETURN r").result_set:
            ontology.add_relation(
                Relation.from_graph(relation[0], [x for xs in entities for x in xs])
            )

        return ontology

    def add_entity(self, entity: Entity):
        """
        Adds an entity to the ontology.

        Parameters:
            entity: The entity object to be added.
        """
        self.entities.append(entity)

    def add_relation(self, relation: Relation):
        """
        Adds a relation to the ontology.

        Args:
            relation (Relation): The relation to be added.
        """
        self.relations.append(relation)

    def to_json(self):
        """
        Converts the ontology object to a JSON representation.

        Returns:
            A dictionary representing the ontology object in JSON format.
        """
        return {
            "entities": [entity.to_json() for entity in self.entities],
            "relations": [relation.to_json() for relation in self.relations],
        }

    def merge_with(self, o: "Ontology"):
        """
        Merges the given ontology `o` with the current ontology.

        Args:
            o (Ontology): The ontology to merge with.

        Returns:
            The merged ontology.
        """
        # Merge entities
        for entity in o.entities:
            if entity.label not in [n.label for n in self.entities]:
                # Entity does not exist in self, add it
                self.entities.append(entity)
                logger.debug(f"Adding entity {entity.label}")
            else:
                # Entity exists in self, merge attributes
                entity1 = next(n for n in self.entities if n.label == entity.label)
                entity1.merge(entity)

        # Merge relations
        for relation in o.relations:
            if relation.label not in [e.label for e in self.relations]:
                # Relation does not exist in self, add it
                self.relations.append(relation)
                logger.debug(f"Adding relation {relation.label}")
            else:
                # Relation exists in self, merge attributes
                relation1 = next(e for e in self.relations if e.label == relation.label)
                relation1.combine(relation)

        return self

    def discard_entities_without_relations(self):
        """
        Discards entities that do not have any relations in the ontology.

        Returns:
            The updated ontology object after discarding entities without relations.
        """
        entities_to_discard = [
            entity.label
            for entity in self.entities
            if all(
                [
                    relation.source.label != entity.label
                    and relation.target.label != entity.label
                    for relation in self.relations
                ]
            )
        ]

        self.entities = [
            entity
            for entity in self.entities
            if entity.label not in entities_to_discard
        ]
        self.relations = [
            relation
            for relation in self.relations
            if relation.source.label not in entities_to_discard
            and relation.target.label not in entities_to_discard
        ]

        if len(entities_to_discard) > 0:
            logger.info(f"Discarded entities: {', '.join(entities_to_discard)}")

        return self

    def discard_relations_without_entities(self):
        """
        Discards relations that have entities not present in the ontology.

        Returns:
            The current instance of the Ontology class.
        """
        relations_to_discard = [
            relation.label
            for relation in self.relations
            if relation.source.label not in [entity.label for entity in self.entities]
            or relation.target.label not in [entity.label for entity in self.entities]
        ]

        self.relations = [
            relation
            for relation in self.relations
            if relation.label not in relations_to_discard
        ]

        if len(relations_to_discard) > 0:
            logger.info(f"Discarded relations: {', '.join(relations_to_discard)}")

        return self

    def validate_entities(self):
        """
        Validates the entities in the ontology.

        This method checks for entities without unique attributes and logs a warning if any are found.

        Returns:
            True if all entities have unique attributes, False otherwise.
        """
        # Check for entities without unique attributes
        entities_without_unique_attributes = [
            entity.label
            for entity in self.entities
            if len(entity.get_unique_attributes()) == 0
        ]
        if len(entities_without_unique_attributes) > 0:
            logger.warn(
                f"""
*** WARNING ***
The following entities do not have unique attributes:
{', '.join(entities_without_unique_attributes)}
"""
            )
            return False
        return True

    def get_entity_with_label(self, label: str):
        """
        Retrieves the entity with the specified label.

        Args:
            label (str): The label of the entity to retrieve.

        Returns:
            The entity with the specified label, or None if not found.
        """
        return next((n for n in self.entities if n.label == label), None)

    def get_relations_with_label(self, label: str):
        """
        Returns a list of relations with the specified label.

        Parameters:
            label (str): The label to search for.

        Returns:
            A list of relations with the specified label.
        """
        return [e for e in self.relations if e.label == label]

    def has_entity_with_label(self, label: str):
        """
        Checks if the ontology has an entity with the given label.

        Parameters:
            label (str): The label to search for.

        Returns:
            True if an entity with the given label exists, False otherwise.
        """
        return any(n.label == label for n in self.entities)

    def has_relation_with_label(self, label: str):
        """
        Checks if the ontology has a relation with the given label.

        Parameters:
            label (str): The label of the relation to check.

        Returns:
            True if a relation with the given label exists, False otherwise.
        """
        return any(e.label == label for e in self.relations)

    def __str__(self) -> str:
        """
        Returns a string representation of the Ontology object.

        The string includes a list of entities and relations in the ontology.

        Returns:
            A string representation of the Ontology object.
        """
        return "Entities:\n\f- {entities}\n\nEdges:\n\f- {relations}".format(
            entities="\n- ".join([str(entity) for entity in self.entities]),
            relations="\n- ".join([str(relation) for relation in self.relations]),
        )

    def save_to_graph(self, graph: Graph):
        """
        Saves the entities and relations to the specified graph.

        Args:
            graph (Graph): The graph to save the entities and relations to.
        """
        for entity in self.entities:
            query = entity.to_graph_query()
            logger.debug(f"Query: {query}")
            graph.query(query)

        for relation in self.relations:
            query = relation.to_graph_query()
            logger.debug(f"Query: {query}")
            graph.query(query)
