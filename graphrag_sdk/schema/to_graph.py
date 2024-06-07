from falkordb import Graph
from .entity import Entity
from .relation import Relation

# Save schema as graph
def schema_to_graph(s, g:Graph) -> None:
    # Validate arguments
    if not isinstance(g, Graph):
        raise Exception("Invalid argument, g is expected to be a Graph")

    # Make sure schema is valid
    s.validate()

    # Clear graph
    try:
        g.delete()
    except:
        pass

    # Create graph entities
    entity_ids = {}
    for e in s.entities:
        # Construct entity creation query
        q = f"""CREATE (n:{e.name})
                SET n = $args
                RETURN ID(n)"""

        # Build query arguments
        args = {}
        for attr in e.attributes:
            args[attr.name] = [attr.type.__name__, attr.desc, attr.unique, attr.mandatory]

        # Run query and get entity id
        id = g.query(q, params={'args': args}).result_set[0][0]

        # Save
        entity_ids[e.name] = id

    # Create graph relations
    for r in s.relations:
        # Construct relationship creation query
        q = f"""MATCH (src), (dest)
                WHERE ID(src) = $src AND ID(dest) = $dest
                CREATE (src)-[:{r.name}]->(dest)"""

        # Build query arguments
        args = {'src': entity_ids[r.src.name], 'dest':entity_ids[r.dest.name]}

        # Run query and get entity id
        g.query(q, params=args)
