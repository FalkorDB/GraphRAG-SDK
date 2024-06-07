from .entity import Entity

# Represents an ontolegy relation
# A relation is composed of:
# 1. name
# 2. src entity
# 3. destination entity
class Relation(object):
    def __init__(self, name:str, src:Entity, dest:Entity):
        # Validate arguments
        if not isinstance(name, str) or name == "":
            raise Exception("Relation name must be a none empty string")
        if not isinstance(src, Entity):
            raise Exception("src must be am entity")
        if not isinstance(dest, Entity):
            raise Exception("dest must be am entity")

        self.src  = src
        self.dest = dest
        self.name = name

    def __str__(self) -> str:
        return f"(:{self.src.name})-[:{self.name}]->(:{self.dest.name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Relation):
            return False

        return (self.src == other.src and
                self.dest == other.dest and
                self.name == other.name)

    def __lt__(self, other):
        return (self.src, self.name, self.dest) < (other.src, other.name, other.dest)

    def __hash__(self) -> int:
        return hash((self.src, self.name, self.dest))
