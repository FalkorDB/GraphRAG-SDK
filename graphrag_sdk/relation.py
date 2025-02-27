import re
import json
import logging
from .attribute import Attribute
from typing import Union, Optional
from falkordb import Node as GraphNode, Edge as GraphEdge
from graphrag_sdk.fixtures.regex import (
    EDGE_LABEL_REGEX,
    NODE_LABEL_REGEX,
    EDGE_REGEX,
)


logger = logging.getLogger(__name__)

class _RelationEntity:
    """
    Represents a relation entity.

    Attributes:
        label (str): The label of the relation entity.
    """

    def __init__(self, label: str):
        """
        Initializes a Relation object with the given label.

        Args:
            label (str): The label of the relation.

        Returns:
            None
        """
        self.label = re.sub(r"([^a-zA-Z0-9_])", "", label)

    @staticmethod
    def from_json(txt: str) -> "_RelationEntity":
        """
        Creates a _RelationEntity object from a JSON string.

        Args:
            txt (str): The JSON string representing the _RelationEntity object.

        Returns:
            _RelationEntity: The created _RelationEntity object.
        """
        txt = txt if isinstance(txt, dict) else json.loads(txt)
        return _RelationEntity(txt.get("label", txt))

    def to_json(self) -> dict:
        """
        Converts the _RelationEntity object to a JSON string.

        Returns:
            str: The JSON string representation of the _RelationEntity object.
        """
        return {"label": self.label}

    def __str__(self) -> str:
        """
        Returns a string representation of the Relation object.
        
        The string representation includes the label of the Relation object.
        
        Returns:
            str: The string representation of the Relation object.
        """
        return f"(:{self.label})"


class Relation:
    """
    Represents a relation between two entities in a graph.

    Attributes:
        label (str): The label of the relation.
        source (Union[_RelationEntity, str]): The source entity of the relation.
        target (Union[_RelationEntity, str]): The target entity of the relation.
        attributes (Optional[list[Attribute]]): The attributes associated with the relation.

    Methods:
        from_graph(relation: GraphEdge, entities: list[GraphNode]) -> Relation:
            Creates a Relation object from a graph edge and a list of graph nodes.
        from_json(txt: Union[dict, str]) -> Relation:
            Creates a Relation object from a JSON string or dictionary.
        from_string(txt: str) -> Relation:
            Creates a Relation object from a string representation.
        to_json() -> dict:
            Converts the Relation object to a JSON dictionary.
        combine(relation2: "Relation") -> Relation:
            Combines the attributes of another Relation object with this Relation object.
        to_graph_query() -> str:
            Generates a Cypher query string for creating the relation in a graph database.
        __str__() -> str:
            Returns a string representation of the Relation object.
    """

    def __init__(
        self,
        label: str,
        source: Union[_RelationEntity, str],
        target: Union[_RelationEntity, str],
        attributes: Optional[list[Attribute]] = None,
    ):
        """
        Initializes a Relation object.

        Args:
            label (str): The label of the relation.
            source (Union[_RelationEntity, str]): The source entity of the relation.
            target (Union[_RelationEntity, str]): The target entity of the relation.
            attributes (Optional[list[Attribute]]): The attributes associated with the relation. Defaults to None.
        """
        attributes = attributes or []
        if isinstance(source, str):
            source = _RelationEntity(source)
        if isinstance(target, str):
            target = _RelationEntity(target)

        assert isinstance(label, str), "Label must be a string"
        assert isinstance(source, _RelationEntity), "Source must be an EdgeNode"
        assert isinstance(target, _RelationEntity), "Target must be an EdgeNode"
        assert isinstance(attributes, list), "Attributes must be a list"

        self.label = re.sub(r"([^a-zA-Z0-9_])", "", label.upper())
        self.source = source
        self.target = target
        self.attributes = attributes

    @staticmethod
    def from_graph(relation: GraphEdge, entities: list[GraphNode]) -> "Relation":
        """
        Creates a Relation object from a graph edge and a list of graph nodes.

        Args:
            relation (GraphEdge): The graph edge representing the relation.
            entities (list[GraphNode]): The list of graph nodes representing the entities.

        Returns:
            Relation: The created Relation object.
        """
        logger.debug(f"Relation.from_graph: {relation}")
        return Relation(
            relation.relation,
            _RelationEntity(
                next(n.labels[0] for n in entities if n.id == relation.src_node)
            ),
            _RelationEntity(
                next(n.labels[0] for n in entities if n.id == relation.dest_node)
            ),
            [
                Attribute.from_string(f"{attr}:{relation.properties[attr]}")
                for attr in relation.properties
            ],
        )

    @staticmethod
    def from_json(txt: Union[dict, str]) -> "Relation":
        """
        Creates a Relation object from a JSON string or dictionary.

        Args:
            txt (Union[dict, str]): The JSON string or dictionary representing the relation.

        Returns:
            Relation: The created Relation object.
        """
        txt = txt if isinstance(txt, dict) else json.loads(txt)
        return Relation(
            txt["label"],
            _RelationEntity.from_json(txt["source"]),
            _RelationEntity.from_json(txt["target"]),
            (
                [Attribute.from_json(attr) for attr in txt["attributes"]]
                if "attributes" in txt
                else []
            ),
        )

    @staticmethod
    def from_string(txt: str) -> "Relation":
        """
        Creates a Relation object from a string representation.

        Args:
            txt (str): The string representation of the relation.

        Returns:
            Relation: The created Relation object.
        """
        label = re.search(EDGE_LABEL_REGEX, txt).group(0).strip()
        source = re.search(NODE_LABEL_REGEX, txt).group(0).strip()
        target = re.search(NODE_LABEL_REGEX, txt).group(1).strip()
        relation = re.search(EDGE_REGEX, txt).group(0)
        attributes = (
            relation.split("{")[1].split("}")[0].strip().split(",")
            if "{" in relation
            else []
        )

        return Relation(
            label,
            _RelationEntity(source),
            _RelationEntity(target),
            [Attribute.from_string(attr) for attr in attributes],
        )

    def to_json(self) -> dict:
        """
        Converts the Relation object to a JSON dictionary.

        Returns:
            dict: The JSON dictionary representing the Relation object.
        """
        return {
            "label": self.label,
            "source": self.source.to_json(),
            "target": self.target.to_json(),
            "attributes": [attr.to_json() for attr in self.attributes],
        }

    def combine(self, relation2: "Relation") -> "Relation":
        """
        Overwrites attributes of self with attributes of relation2.

        Args:
            relation2 (Relation): The Relation object whose attributes will be combined.

        Returns:
            Relation: The combined Relation object.
        """
        if self.label != relation2.label:
            raise Exception("Relations must have the same label to be combined")

        for attr in relation2.attributes:
            if attr.name not in [a.name for a in self.attributes]:
                logger.debug(f"Adding attribute {attr.name} to relation {self.label}")
                self.attributes.append(attr)

        return self

    def to_graph_query(self) -> str:
        """
        Generates a Cypher query string for creating the relation in a graph database.

        Returns:
            str: The Cypher query string.
        """
        return f"MATCH (s:{self.source.label}) MATCH (t:{self.target.label}) MERGE (s)-[r:{self.label} {{{', '.join([str(attr) for attr in self.attributes])}}}]->(t) RETURN r"

    def __str__(self) -> str:
        """
        Returns a string representation of the Relation object.

        Returns:
            str: The string representation of the Relation object.
        """
        return f"{self.source}-[:{self.label} {{{', '.join([str(attr) for attr in self.attributes])}}}]->{self.target}"
