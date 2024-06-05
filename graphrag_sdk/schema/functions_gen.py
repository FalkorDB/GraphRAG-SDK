from .schema import Schema
from .entity import Entity
from .relation import Relation

def relation_to_create_func(r:Relation):
    src_lbl = r.src.name
    dest_lbl = r.dest.name
    func_name = r.name

    func = f"def {r.name}(args):\n"       # def MovieStream(args):
    func += "\targs = remove_none_values(args)\n"   # args = remove_none_values(args)

    # extract required src arguments
    src_attrs = r.src.unique_attributes()

    # extract required dest arguments
    func += "\tq = \"MERGE (s:" + src_lbl + " {"
    func += ", ".join([f"{attr.name}: " + f"${attr.name}" for attr in src_attrs])
    func += "}) "

    # extract required src arguments
    dest_attrs = r.dest.unique_attributes()

    func += "MERGE (d:" + dest_lbl + " {"
    func += ", ".join([f"{attr.name}: " + f"${attr.name}" for attr in dest_attrs])
    func += "}) "

    func += "MERGE (s)-[r:" + r.name + "]->(d)\"\n"

    # construct query params
    func += "\tparams = {"
    func += ", ".join([f"'{attr.name}': args['{attr.name}']" for attr in src_attrs])
    func += ", "
    func += ", ".join([f"'{attr.name}': args['{attr.name}']" for attr in dest_attrs])
    func += "}\n"

    func += "\tg.query(q, params)\n"

    return (func_name, func)

def entity_to_create_func(e:Entity):
    func_name = f"Create_{e.name}"  # Create_Actor
    func = f"def {func_name}(args):\n" # def Create_Actor(args):

    func += "\targs = remove_none_values(args)\n"
    func += "\tquery = \"MERGE (n:" + e.name + " {" # query = "MERGE (n:Actor {

    unique_attributes = e.unique_attributes()

    func += ", ".join([f"{attr.name}: ${attr.name}" for attr in unique_attributes]) + "})"
    func += " SET n += $args\"\n"

    func += "\tparams = {"
    func += ", ".join([f"'{attr.name}': args['{attr.name}']" for attr in unique_attributes])
    func += ", 'args': args}\n"

    func += "\tg.query(query, params)\n"

    return (func_name, func)

def schema_to_functions(s:Schema):
    if not isinstance(s, Schema):
        raise Exception("Invalid argument, expecting a Schema object")

    functions = []
    for e in s.entities:
        func = entity_to_create_func(s.entities[e])
        functions.append(func)

    for r in s.relations:
        func = relation_to_create_func(s.relations[r])
        functions.append(func)

    return functions
