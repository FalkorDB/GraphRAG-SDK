from falkordb import Graph
from .entity import Entity
from .relation import Relation

import builtins

# Create a schema from an ontology graph
def schema_from_graph(s, g:Graph):
    # Validate arguments
    if not isinstance(g, Graph):
        raise Exception("Invalid argument, g is expected to be a Graph")

    # Fetch entities
    q = "MATCH (n) RETURN n"
    res = g.query(q).result_set

    for row in res:
        n = row[0]
        name = n.labels[0]

        e = s.add_entity(name)
        for prop in n.properties:
            name = prop
            prop = n.properties[prop]

            type      = getattr(builtins, prop[0])
            desc      = prop[1]
            unique    = prop[2]
            mandatory = prop[3]

            e.add_attribute(name, type, desc, unique, mandatory)

    # Fetch relations
    q = "MATCH (src)-[e]->(dest) RETURN src, e, dest"
    res = g.query(q).result_set

    for row in res:
        src  = row[0].labels[0]
        r    = row[1].relation
        dest = row[2].labels[0]

        src  = s.get_entity(src)
        dest = s.get_entity(dest)

        s.add_relation(r, src, dest)

    # Make sure schema is valid
    s.validate()
