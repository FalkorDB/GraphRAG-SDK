import json
from graphrag_sdk.fixtures.regex import *
import logging
import re

logger = logging.getLogger(__name__)


class AttributeType:
    """
    Represents the types of attributes in the system.
    """

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"

    @staticmethod
    def from_string(txt: str):
        """
        Converts a string representation of an attribute type to its corresponding AttributeType value.

        Args:
            txt (str): The string representation of the attribute type.

        Returns:
            AttributeType: The corresponding AttributeType value.

        Raises:
            Exception: If the provided attribute type is invalid.
        """
        if txt.lower() == AttributeType.STRING:
            return AttributeType.STRING
        if txt.lower() == AttributeType.NUMBER:
            return AttributeType.NUMBER
        if txt.lower() == AttributeType.BOOLEAN:
            return AttributeType.BOOLEAN
        raise Exception(f"Invalid attribute type: {txt}")


class Attribute:
    """ Represents an attribute of an entity or relation in the ontology.

        Args:
            name (str): The name of the attribute.
            attr_type (AttributeType): The type of the attribute.
            unique (bool): Whether the attribute is unique.
            required (bool): Whether the attribute is required.

        Examples:
            >>> attr = Attribute("name", AttributeType.STRING, True, True)
            >>> print(attr)
            name: "string!*"
    """

    def __init__(
        self, name: str, attr_type: AttributeType, unique: bool, required: bool = False
    ):
        """
        Initialize a new Attribute object.

        Args:
            name (str): The name of the attribute.
            attr_type (AttributeType): The type of the attribute.
            unique (bool): Indicates whether the attribute should be unique.
            required (bool, optional): Indicates whether the attribute is required. Defaults to False.
        """
        self.name = re.sub(r"([^a-zA-Z0-9_])", "_", name)
        self.type = attr_type
        self.unique = unique
        self.required = required

    @staticmethod
    def from_json(txt: str | dict):
        """
        Creates an Attribute object from a JSON string or dictionary.

        Args:
            txt (str | dict): The JSON string or dictionary representing the Attribute.

        Returns:
            Attribute: The created Attribute object.

        """
        txt = txt if isinstance(txt, dict) else json.loads(txt)

        return Attribute(
            txt["name"],
            AttributeType.from_string(txt["type"]),
            txt["unique"],
            txt["required"] if "required" in txt else False,
        )

    @staticmethod
    def from_string(txt: str):
        """
        Parses an attribute from a string.
        The "!" symbol indicates that the attribute is unique.
        The "*" symbol indicates that the attribute is required

        Args:
            txt (str): The string to parse.

        Returns:
            Attribute: The parsed attribute.

        Examples:
            >>> attr = Attribute.from_string("name:string!*")
            >>> print(attr.name)
            name

        Raises:
            Exception: If the attribute type is invalid.
        """
        name = txt.split(":")[0].strip()
        attr_type = txt.split(":")[1].split("!")[0].split("*")[0].strip()
        unique = "!" in txt
        required = "*" in txt

        if attr_type not in [
            AttributeType.STRING,
            AttributeType.NUMBER,
            AttributeType.BOOLEAN,
        ]:
            raise Exception(f"Invalid attribute type: {attr_type}")

        return Attribute(name, AttributeType.from_string(attr_type), unique, required)

    def to_json(self):
        """
        Converts the attribute object to a JSON representation.

        Returns:
            dict: A dictionary representing the attribute object in JSON format.
                The dictionary contains the following keys:
                - "name": The name of the attribute.
                - "type": The type of the attribute.
                - "unique": A boolean indicating whether the attribute is unique.
                - "required": A boolean indicating whether the attribute is required.
        """
        return {
            "name": self.name,
            "type": self.type,
            "unique": self.unique,
            "required": self.required,
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the Attribute object.

        The string representation includes the attribute name, type, uniqueness, and requirement status.

        Returns:
            str: A string representation of the Attribute object.
        """
        return f"{self.name}: \"{self.type}{'!' if self.unique else ''}{'*' if self.required else ''}\""
