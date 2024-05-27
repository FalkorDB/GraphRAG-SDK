from .entity import Entity
from .schema import Schema
from .relation import Relation

def type_mapping(t) -> str:
    if t not in [str, int, float, bool]:
        raise Exception(f"Unsupported type: {t.__name__}")

    if t == str:
        return "string"
    elif t == bool:
        return "boolean"
    else:
        return "number"

def relation_to_tool(r:Relation):
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "directed",
    #         "description": "Form a connection between a director and a movie",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "src": {"type": "string", "description": "director's name"},
    #                 "dest": {"type": "string", "description": "movie's title"}
    #             },
    #             "required": ["src", "dest"]
    #         }
    #     }
    # }

    src  = r.src
    dest = r.dest

    tool = {}
    tool['type'] = "function"
    tool['function'] = {}
    tool['function']['name'] = r.name
    tool['function']['description'] = "Form a connection between " + src.name + " and " + dest.name

    tool['function']['parameters'] = {}
    tool['function']['parameters']['type'] = "object"
    tool['function']['parameters']['properties'] = {}

    required = []

    # process edge's src
    # see if src node has a unique attribute
    attrs = src.unique_attributes()
    for attr in attrs:
        tool['function']['parameters']['properties'][attr.name] = {}
        tool['function']['parameters']['properties'][attr.name]['type'] = type_mapping(attr.type)
        if attr.desc is not None:
            tool['function']['parameters']['properties'][attr.name]['description'] = attr.desc
        else:
            tool['function']['parameters']['properties'][attr.name]['description'] = src.name + "'s " + attr.name
        required.append(attr.name)

    # process edge's destination
    # see if dest node has a unique attribute
    attrs = dest.unique_attributes()
    for attr in attrs:
        tool['function']['parameters']['properties'][attr.name] = {}
        tool['function']['parameters']['properties'][attr.name]['type'] = type_mapping(attr.type)
        if attr.desc is not None:
            tool['function']['parameters']['properties'][attr.name]['description'] = attr.desc
        else:
            tool['function']['parameters']['properties'][attr.name]['description'] = dest.name + "'s " + attr.name
        required.append(attr.name)

    tool['function']['parameters']['required'] = required

    return tool

def entity_to_tool(e:Entity):
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "directed",
    #         "description": "Form a connection between a director and a movie",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "src": {"type": "string", "description": "director's name"},
    #                 "dest": {"type": "string", "description": "movie's title"}
    #             },
    #             "required": ["src", "dest"]
    #         }
    #     }
    # }

    tool = {}
    tool['type'] = "function"
    tool['function'] = {}

    tool['function']['name'] = "Create_" + e.name
    tool['function']['description'] = "Create a new " + e.name

    tool['function']['parameters'] = {}
    tool['function']['parameters']['type'] = "object"
    tool['function']['parameters']['properties'] = {}

    unique_attributes = []
    for attr in e.attributes:
        attribute_name = attr.name
        attribute_type = type_mapping(attr.type)
        attribute_unique = attr.unique

        tool['function']['parameters']['properties'][attribute_name] = {}
        tool['function']['parameters']['properties'][attribute_name]['type'] = attribute_type
        if attr.desc is None:
            tool['function']['parameters']['properties'][attribute_name]['description'] = f"{e.name}'s {attr.name}"
        else:
            tool['function']['parameters']['properties'][attribute_name]['description'] = attr.desc

        if attribute_unique:
            unique_attributes.append(attribute_name)

    tool['function']['parameters']['required'] = unique_attributes

    return tool

# Create OpenAI tools from schema
def schema_to_tools(s:Schema):
    if not isinstance(s, Schema):
        raise Exception("Invalid argument, expecting a Schema object")

    tools = []
    for e in s.entities:
        tool = entity_to_tool(s.entities[e])
        tools.append(tool)

    for r in s.relations:
        tool = relation_to_tool(s.relations[r])
        tools.append(tool)

    return tools
