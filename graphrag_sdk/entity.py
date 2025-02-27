import re
import json
import logging
from typing import Union
from .attribute import Attribute
from falkordb import Node as GraphNode


logger = logging.getLogger(__name__)
descriptionKey = "__description__"

class Entity:
    """
    Represents an entity in the knowledge graph.

    Attributes:
        label (str): The label of the entity.
        attributes (list[Attribute]): The attributes associated with the entity.
        description (str): The description of the entity (optional).

    Methods:
        from_graph(entity: GraphNode) -> Entity: Creates an Entity object from a GraphNode object.
        from_json(txt: Union[dict, str]) -> Entity: Creates an Entity object from a JSON string or dictionary.
        to_json() -> dict: Converts the Entity object to a JSON dictionary.
        merge(entity2: Entity) -> Entity: Overwrites attributes of self with attributes of entity2.
        get_unique_attributes() -> list[Attribute]: Returns a list of unique attributes of the entity.
        to_graph_query() -> str: Generates a Cypher query to merge the entity in the graph.
    """

    def __init__(self, label: str, attributes: list[Attribute], description: str = ""):
        """
        Initialize a new Entity object.

        Args:
            label (str): The label of the entity.
            attributes (list[Attribute]): A list of Attribute objects associated with the entity.
            description (str, optional): The description of the entity. Defaults to "".
        """
        self.label = re.sub(r"([^a-zA-Z0-9_])", "", label)
        self.attributes = attributes
        self.description = description

    @staticmethod
    def from_graph(entity: GraphNode) -> "Entity":
        """
        Converts a GraphNode object to an Entity object.

        Args:
            entity (GraphNode): The GraphNode object to convert.

        Returns:
            Entity: The converted Entity object.

        """
        logger.debug(f"Entity.from_graph: {entity}")
        return Entity(
            entity.labels[0],
            [
                Attribute.from_string(f"{attr}:{entity.properties[attr]}")
                for attr in entity.properties
                if attr != descriptionKey
            ],
            entity.properties.get(descriptionKey, ""),
        )

    @staticmethod
    def from_json(txt: Union[dict, str]) -> "Entity":
        """
        Create an Entity object from a JSON representation.

        Args:
            txt (Union[dict, str]): The JSON representation of the Entity. It can be either a dictionary or a string.

        Returns:
            Entity: The Entity object created from the JSON representation.

        """
        txt = txt if isinstance(txt, dict) else json.loads(txt)
        return Entity(
            txt["label"],
            [Attribute.from_json(attr) for attr in (txt.get("attributes", []))],
            txt.get("description", ""),
        )

    def to_json(self) -> dict:
        """
        Convert the entity object to a JSON representation.

        Returns:
            dict: A dictionary representing the entity object in JSON format.
                The dictionary contains the following keys:
                - "label": The label of the entity.
                - "attributes": A list of attribute objects converted to JSON format.
                - "description": The description of the entity.
        """
        return {
            "label": self.label,
            "attributes": [attr.to_json() for attr in self.attributes],
            "description": self.description,
        }

    def merge(self, entity2: "Entity") -> "Entity":
        """Overwrite attributes of self with attributes of entity2.

        Args:
            entity2 (Entity): The entity to merge with self.

        Raises:
            Exception: If the entities have different labels.

        Returns:
            Entity: The merged entity.
        """
        if self.label != entity2.label:
            raise Exception("Entities must have the same label to be combined")

        for attr in entity2.attributes:
            if attr.name not in [a.name for a in self.attributes]:
                logger.debug(f"Adding attribute {attr.name} to entity {self.label}")
                self.attributes.append(attr)

        return self

    def get_unique_attributes(self) -> list[Attribute]:
        """
        Returns a list of unique attributes for the entity.

        Returns:
            list: A list of attributes that are marked as unique.
        """
        return [attr for attr in self.attributes if attr.unique]

    def to_graph_query(self) -> str:
        """
        Generates a Cypher query string for creating or updating a node in a graph database.

        Returns:
            str: The Cypher query string.
        """
        unique_attributes = ", ".join(
            [str(attr) for attr in self.attributes if attr.unique]
        )
        non_unique_attributes = ", ".join(
            [str(attr) for attr in self.attributes if not attr.unique]
        )
        if self.description:
            non_unique_attributes += f"{', ' if len(non_unique_attributes) > 0 else ''} {descriptionKey}: '{self.description}'"
        return f"MERGE (n:{self.label} {{{unique_attributes}}}) SET n += {{{non_unique_attributes}}} RETURN n"

    def __str__(self) -> str:
        """
        Returns a string representation of the Entity object.

        The string representation includes the label of the entity and its attributes.

        Returns:
            str: The string representation of the Entity object.
        """
        return (
            f"(:{self.label} {{{', '.join([str(attr) for attr in self.attributes])}}})"
        )
