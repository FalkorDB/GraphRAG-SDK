from falkordb import Graph
from graphrag_sdk.source import AbstractSource
from .entity import Entity
from .relation import Relation
from .encoder import SchemaEncoder
from .decoder import SchemaDecoder
from .to_graph import schema_to_graph
from .from_graph import schema_from_graph
from .auto_detect import schema_auto_detect

class Schema(object):
    """
    Knowledge Graph schema (ontology)
    The schema captures entities types in addition to their attributes
    and the relationship types between the them
    """

    def __init__(self):
        """
        Initialize Schema
        """

        self.entities = set([])
        self.relations = set([])

    def __eq__(self, other) -> bool:
        if not isinstance(other, Schema):
            return False

        if self.entities != other.entities:
            return False

        if self.relations != other.relations:
            return False

        return True


    def add_entity(self, name:str) -> Entity:
        """
        Add a new entity to schema

         Parameters:
             name (str): Entity name.

        Returns:
            entity
        """

        # Validate arguments
        if not isinstance(name, str) or name == "":
            raise Exception(f"Invalid name argument, expecting none empty string")
        if name in self.entities:
            raise Exception(f"Entity {name} already exists")

        e = Entity(name)
        self.entities.add(e)

        return e

    def get_entity(self, name:str) -> Entity | None:
        """
        Get an entity from schema

        Returns:
            entity
        """

        if not isinstance(name, str) or name == "":
            raise Exception("Invalid argument, name should be a none empty string")

        for e in self.entities:
            if e.name == name:
                return e

        return None

    def add_relation(self, name:str, src:Entity, dest:Entity) -> Relation:
        """
        Add a relationship to schema

        Returns:
            relation
        """

        # Validate arguments
        if not isinstance(src, Entity) and not isinstance(src, str):
            raise Exception(f"Invalid argument, src is expected to be either a string or an Entity")
        if not isinstance(dest, Entity) and not isinstance(dest, str):
            raise Exception(f"Invalid argument, dest is expected to be either a string or an Entity")
        if not isinstance(name, str) or name == "":
            raise Exception(f"Invalid name argument, expecting none empty string")

        if isinstance(src, str) and src == "":
            raise Exception(f"Invalid src argument, expecting a non empty string")
        if isinstance(dest, str) and dest == "":
            raise Exception(f"Invalid dest argument, expecting a non empty string")

        if name in self.relations:
            raise Exception(f"Relation {name} already exists")

        # Both src and dest must have at least one unique attribute
        if len(src.unique_attributes()) == 0:
            raise Exception(f"{src.name} must have at least one unique attribute")
        if len(dest.unique_attributes()) == 0:
            raise Exception(f"{dest.name} must have at least one unique attribute")

        # Create relation and add it to schema
        r = Relation(name, src, dest)
        self.relations.add(r)

        return r

    def validate(self) -> bool:
        """
        Validate schema

        Returns:
            True if schema is valid
        """

        # Make sure each Relation reference existing Entities
        # And each referenced entity has at least one unique attribute
        for r in self.relations:
            src = r.src
            dest = r.dest

            if src not in self.entities:
                raise Exception(f"Relation {r.name} reference a none existing entity {src}")
            if dest not in self.entities:
                raise Exception(f"Relation {r.name} reference a none existing entity {dest}")
            if len(src.unique_attributes()) == 0:
                raise Exception(f"Relation {r.name} src endpoint {src.name} is missing a unique attribute")
            if len(dest.unique_attributes()) == 0:
                raise Exception(f"Relation {r.name} src endpoint {dest.name} is missing a unique attribute")

        return True

    def save_graph(self, g:Graph) -> None:
        """
        Save schema as an ontology graph

        Parameters:
            g (Graph): graph to create
        """

        schema_to_graph(self, g)

    def to_JSON(self) -> str:
        """
        Encode schema to JSON string

        Returns:
            JSON representation of schema
        """

        encoder = SchemaEncoder()
        return encoder.to_JSON(self)

    @classmethod
    def from_graph(cls, g:Graph) -> 'Schema':
        """
        Create a schema from an ontology graph

        Parameters:
            g (Graph): ontology graph to load

         Returns:
             Schema: schema
        """

        s = cls()
        schema_from_graph(s, g)
        return s

    @classmethod
    def auto_detect(cls, sources: list[AbstractSource], model="gpt-3.5-turbo-0125") -> 'Schema':
        """
        Auto detect schema from sources

        Parameters:
            sources (list[AbstractSource]): list of sources to extract schema from
            model (str): OpenAI model to use

         Returns:
             Schema: schema
        """

        s = cls()
        schema_auto_detect(s, sources, model)
        return s

    @classmethod
    def from_JSON(cls, json:str) -> 'Schema':
        """
        Create a schema from JSON

        Parameters:
            json (str): the output from to_JSON

         Returns:
             Schema: schema
        """

        s = cls()
        decoder = SchemaDecoder()
        decoder.from_JSON(s, json)
        return s
