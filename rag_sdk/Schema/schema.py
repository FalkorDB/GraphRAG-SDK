from falkordb import Graph
from .entity import Entity
from .relation import Relation
from .encoder import SchemaEncoder
from .decoder import SchemaDecoder
from .to_graph import schema_to_graph
from .from_graph import schema_from_graph
from .auto_detect import schema_auto_detect

# Knowledge Graph schema
# The schema captures graph entities in addition to their attributes
# and the relationship types between the entities
class Schema(object):
    def __init__(self):
        self.entities = {}
        self.relations = {}

    def __eq__(self, other) -> bool:
        if not isinstance(other, Schema):
            return False

        if self.entities != other.entities:
            return False

        if self.relations != other.relations:
            return False

        return True

    # Add a new entity to schema
    def add_entity(self, name:str):
        # Validate arguments
        if not isinstance(name, str) or name == "":
            raise Exception(f"Invalid name argument, expecting none empty string")
        if name in self.entities:
            raise Exception(f"Entity {name} already exists")

        e = Entity(name)
        self.entities[name] = e
        return e

    # Get an entity from schema
    def get_entity(self, name:str) -> Entity | None:
        if not isinstance(name, str) or name == "":
            raise Exception("Invalid argument, name should be a none empty string")

        if name in self.entities:
            return self.entities[name]
        else:
            return None

    # Add a new relation to schema
    def add_relation(self, name:str, src:Entity, dest:Entity):
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

        # fetch entities
        if isinstance(src, str):
            src = self.entities[src]
        if isinstance(dest, str):
            dest = self.entities[dest]

        # Both src and dest must have at least one unique attribute
        if len(src.unique_attributes()) == 0:
            raise Exception(f"{src.name} must have at least one unique attribute")
        if len(dest.unique_attributes()) == 0:
            raise Exception(f"{dest.name} must have at least one unique attribute")

        # Create relation and add it to schema
        r = Relation(name, src, dest)
        self.relations[name] = r

        return r

    # Validate schema
    # Make sure schema is valid
    def validate(self) -> bool:
        # Make sure each Relation reference existing Entities
        # And each referenced entity has at least one unique attribute
        for r in self.relations:
            r = self.relations[r]
            src = r.src.name
            dest = r.dest.name
            if src not in self.entities:
                raise Exception(f"Relation {r.name} reference a none existing entity {src}")
            if dest not in self.entities:
                raise Exception(f"Relation {r.name} reference a none existing entity {dest}")

            src = self.entities[src]
            if len(src.unique_attributes()) == 0:
                raise Exception(f"Relation {r.name} src endpoint {src.name} is missing a unique attribute")

            dest = self.entities[dest]
            if len(dest.unique_attributes()) == 0:
                raise Exception(f"Relation {r.name} src endpoint {dest.name} is missing a unique attribute")

        return True

    # Save schema as an ontology graph
    def save_graph(self, g:Graph):
        schema_to_graph(self, g)

    # encode schema to JSON string
    def to_JSON(self) -> str:
        encoder = SchemaEncoder()
        return encoder.to_JSON(self)

    @classmethod
    # Create a schema from an ontology graph
    def from_graph(cls, g:Graph):
        s = cls()
        schema_from_graph(s, g)
        return s

    @classmethod
    # Auto detect schema from sources
    def auto_detect(cls, sources, model="gpt-3.5-turbo-0125") -> 'Schema':
        s = cls()
        schema_auto_detect(s, sources, model)
        return s

    @classmethod
    # Create a schema from JSON
    def from_JSON(cls, json:str) -> 'Schema':
        s = cls()
        decoder = SchemaDecoder()
        decoder.from_JSON(s, json)
        return s
