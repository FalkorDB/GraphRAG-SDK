import builtins
from .entity import Entity
from json import JSONDecoder

# Decode Schema from JSON
class SchemaDecoder():
    def from_JSON(self, schema, json:str):
        # Validate argument
        if not isinstance(json, str) or json == "":
            raise Exception("Unable to decode schema, make sure data is a none empty string")

        return self.decode(schema, JSONDecoder().decode(json))

    # Create Schema from dict
    def decode(self, schema, data:dict):
        if not isinstance(data, dict):
            raise Exception("Unable to decode schema, make sure data is a dict")

        # Validate JSON schema structure
        if 'entities' not in data or 'relations' not in data:
            raise Exception("Invalid schema format")

        entities  = data['entities']
        relations = data['relations']

        # Decode entities
        for e in entities:
            name, attributes = self._entity_decode(e)
            e = schema.add_entity(name)
            for attr in attributes:
                e.add_attribute(attr['name'], attr['type'], attr['desc'], attr['unique'], attr['mandatory'])

        # Decode relations
        for r in relations:
            # Validate relation
            if not isinstance(r, dict):
                    raise Exception("Invalid argument, expecting dict")
            if 'name' not in r:
                raise Exception("Invalid Relation format, missing 'name'")
            if 'src' not in r:
                raise Exception("Invalid Relation format, missing 'src'")
            if 'dest' not in r:
                raise Exception("Invalid Relation format, missing 'dest'")

            src  = r['src']
            dest = r['dest']
            name = r['name']

            src  = schema.get_entity(src)
            dest = schema.get_entity(dest)

            if src is None:
                raise Exception(f"Missing src entity {r['src']}")
            if dest is None:
                raise Exception(f"Missing dest entity {r['dest']}")

            schema.add_relation(name, src, dest)

        return schema

    # Load Entity from dict
    # see entityEncode generated structure
    def _entity_decode(self, raw: dict):
        if not isinstance(raw, dict):
            raise Exception("Invalid argument, expecting dict")
        if 'name' not in raw:
            raise Exception("Invalid format, missing 'name'")
        if 'attributes' not in raw:
            raise Exception("Invalid format, missing 'attributes'")

        name = raw['name']
        attributes = raw['attributes']

        for attr in attributes:
            if 'name' not in attr or 'type' not in attr or 'unique' not in attr:
                raise Exception("Invalid format, attribute must include a 'name', 'type' and 'unique' fields")

            attr['type'] = getattr(builtins, attr['type'])

        return name, attributes
