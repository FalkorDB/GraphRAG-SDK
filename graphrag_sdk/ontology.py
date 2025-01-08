import json
import logging
import graphrag_sdk
from .entity import Entity

from falkordb import Graph
from typing import Optional
from falkordb import FalkorDB
from .relation import Relation
from typing import List, Tuple
from .attribute import Attribute
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.models import GenerativeModel


logger = logging.getLogger(__name__)

nodes_query = """
MATCH (n)
WITH DISTINCT labels(n) AS labels, n
RETURN DISTINCT labels, [k in keys(n) | [k, typeof(n[k])]]
"""

rel_query = """
MATCH (n)-[r]->(m)
UNWIND labels(n) as src_label
UNWIND labels(m) as dst_label
UNWIND type(r) as rel_type
RETURN DISTINCT {start: src_label, type: rel_type, end: dst_label}, [k in keys(r) | [k, typeof(r[k])]]
"""

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

        Parameters:
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
        hide_progress: bool = False,
    ) -> "Ontology":
        """
        Create an Ontology object from a list of sources.

        Parameters:
            sources (list[AbstractSource]): A list of AbstractSource objects representing the sources.
            boundaries (Optinal[str]): The boundaries for the ontology.
            model (GenerativeModel): The generative model to use.
            hide_progress (bool): Whether to hide the progress bar.

        Returns:
            The created Ontology object.
        """
        step = graphrag_sdk.CreateOntologyStep(
            sources=sources,
            ontology=Ontology(),
            model=model,
            hide_progress=hide_progress,
        )

        return step.run(boundaries=boundaries)

    @staticmethod
    def from_json(txt: dict | str):
        """
        Creates an Ontology object from a JSON representation.

        Parameters:
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

        Parameters:
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
    
    @staticmethod
    def from_kg_graph(name: str,
        host: Optional[str] = "127.0.0.1",# user bring the graph
        port: Optional[int] = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        limit: int = 100,):
        """
        Creates an Ontology object from a given Knowledge Graph.

        Parameters:
            name (Optional[str]): The name of the graph.
            host (str): The host of the graph.
            port (Optional[int]): The port of the graph.
            username (Optional[str]): The username of the graph.
            password (Optional[str]): The password of the graph.

        Returns:
            The Ontology object created from the Knowledge Graph.
        """
        ontology = Ontology()
        db = FalkorDB(host=host, port=port, username=username, password=password)
        graph = db.select_graph(name)

        e_labels = graph.query("CALL db.labels()").result_set
        
        for label in e_labels:
            nodes_attributes = graph.query(f"MATCH (c:{label[0]}) RETURN properties(c) LIMIT {limit}").result_set
            attributes = extract_prop_to_ontology(nodes_attributes)
            ontology.add_entity(Entity(label[0], attributes))

        r_labels = graph.query("CALL db.relationshipTypes()").result_set
        for r_label in r_labels:
            for label_s in e_labels:
                for label_t in e_labels:
                    relation_attributes = graph.query(f"MATCH (:{label_s[0]})-[r:{r_label[0]}]->(:{label_t[0]}) RETURN properties(r) LIMIT {limit}").result_set
                    if relation_attributes:
                        attributes = extract_prop_to_ontology(relation_attributes)
                        ontology.add_relation(Relation(r_label[0], label_s[0], label_t[0], attributes))
    
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

        Parameters:
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

        Parameters:
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

        Parameters:
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

        Parameters:
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
            
def extract_prop_to_ontology(result_set) -> List[Attribute]:
    # Sort by the number of properties
    result_set = sorted(result_set, key=lambda x: len(x[0]))
    
    # Extract attributes
    attr_unique = set()
    attributes = []
    for i, result in enumerate(result_set):
        is_required = (i == 0)  # Only the first set of attributes is required
        # ensure that they are in all the set
        for attr, value in result[0].items():
            if attr in attr_unique:
                continue
            attr_unique.add(attr)
            
            # Determine attribute type
            if isinstance(value, str): # check if the value is a string
                attr_type = "string"
            elif isinstance(value, int) or isinstance(value, float): # check if the value is an integer or float
                attr_type = "number"
            elif isinstance(value, bool): # check if the value is a boolean
                attr_type = "boolean"
            elif isinstance(value, list): # check if the value is a list
                attr_type = "list"
            else:
                continue
            
            # Determine uniqueness
            if is_required:
                values_for_property = [item[0].get(attr) for item in result_set if attr in item[0]] # without loop
                is_unique = len(values_for_property) == len(set(values_for_property))
                
            # Append the attribute
            attributes.append(Attribute(attr, attr_type, is_unique, is_required))

    return attributes
