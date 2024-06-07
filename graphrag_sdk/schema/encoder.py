import json

# Encode Schema into JSON
class SchemaEncoder():
    # encode schema to JSON string
    def to_JSON(self, schema) -> str:
        return json.dumps(self.encode(schema), indent=4)

    # encode schema as dict
    def encode(self, schema) -> dict:
        # Make sure schema is valid
        schema.validate()

        dict = {'entities': [], 'relations': []}

        for e in schema.entities:
            dict['entities'].append(self._encode_entity(e))

        for r in schema.relations:
            dict['relations'].append(self._encode_relation(r))

        return dict

    # Dump Entity into a dict
    # format:
    # {'name': name,
    #  'attributes': [
    #       {'name': attr_name,
    #       'type': attr_type,
    #       'unique': attr_unique},...
    #   ]
    # }
    def _encode_entity(self, e):
        dict = {'name': e.name, 'attributes': []}

        for attr in e.attributes:
            dict['attributes'].append({'name': attr.name, 'type': attr.type.__name__, 'desc': attr.desc, 'unique': attr.unique, 'mandatory': attr.mandatory})

        return dict

    # Dump Relation into dict
    def _encode_relation(self, r):
        return {'name': r.name, 'src': r.src.name, 'dest': r.dest.name}
