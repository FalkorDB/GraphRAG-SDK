import unittest
from falkordb import FalkorDB
from rag_sdk import Source
from rag_sdk.Schema import Schema
from rag_sdk.Schema.tools_gen import schema_to_tools
from rag_sdk.Schema.functions_gen import schema_to_functions

class TestSchema(unittest.TestCase):
    def test_schema(self):
        s = Schema()

        f = s.add_entity('Fighter')
        f.add_attribute('name', str, unique=True)
        f.add_attribute('height', int)

        m = s.add_entity('Match')
        m.add_attribute('date', int, unique=True)
        m.add_attribute('id', int, unique=True)
        m.add_attribute('rounds', int)

        s.add_relation('WON', f, m)
        s.add_relation('LOST', f, m)

        s.validate()

        dump = s.to_JSON()
        s = Schema.from_JSON(dump)

    def test_schema_tools(self):
        s = Schema()

        f = s.add_entity('Fighter')
        f.add_attribute('name', str, unique=True)
        f.add_attribute('height', int, desc="Fighter's height in cm")

        m = s.add_entity('Match')
        m.add_attribute('date', int, unique=True, desc="Match date in UNIX timestamp")
        m.add_attribute('id', int, unique=True, desc="Fight number")
        m.add_attribute('rounds', int, desc="Number of rounds, either 3 or 5")

        s.add_relation('WON', f, m)
        s.add_relation('LOST', f, m)

        tools = schema_to_tools(s)
        self.assertEqual(len(tools), 4)

        tool = {'type': 'function',
            'function':
                {'name': 'Create_Fighter',
                    'description': 'Create a new Fighter',
                    'parameters': {'type': 'object', 'properties':
                        {'name': {'type': 'string', 'description': "Fighter's name"},
                            'height': {'type': 'number', 'description': "Fighter's height in cm"}
                        },
                'required': ['name']}
            }
        }
        self.assertIn(tool, tools)

    def test_schema_funs(self):
        s = Schema()

        f = s.add_entity('Fighter')
        f.add_attribute('name', str, unique=True)
        f.add_attribute('height', int, desc="Fighter's height in cm")

        m = s.add_entity('Match')
        m.add_attribute('date', int, unique=True, desc="Match date in UNIX timestamp")
        m.add_attribute('id', int, unique=True, desc="Fight number")
        m.add_attribute('rounds', int, desc="Number of rounds, either 3 or 5")

        s.add_relation('WON', f, m)
        s.add_relation('LOST', f, m)

        funcs = schema_to_functions(s)
        self.assertEqual(len(funcs), 4)

        func = ('Create_Fighter', 'def Create_Fighter(args):\n\targs = remove_none_values(args)\n\tquery = "MERGE (n:Fighter {name: $name}) SET n += $args"\n\tparams = {\'name\': args[\'name\'], \'args\': args}\n\tg.query(query, params)\n')
        self.assertIn(func, funcs)

        func = ('Create_Match', 'def Create_Match(args):\n\targs = remove_none_values(args)\n\tquery = "MERGE (n:Match {date: $date, id: $id}) SET n += $args"\n\tparams = {\'date\': args[\'date\'], \'id\': args[\'id\'], \'args\': args}\n\tg.query(query, params)\n')
        self.assertIn(func, funcs)

        func = ('WON', 'def WON(args):\n\targs = remove_none_values(args)\n\tq = "MERGE (s:Fighter {name: $name}) MERGE (d:Match {date: $date, id: $id}) MERGE (s)-[r:WON]->(d)"\n\tparams = {\'name\': args[\'name\'], \'date\': args[\'date\'], \'id\': args[\'id\']}\n\tg.query(q, params)\n')
        self.assertIn(func, funcs)

        func = ('LOST', 'def LOST(args):\n\targs = remove_none_values(args)\n\tq = "MERGE (s:Fighter {name: $name}) MERGE (d:Match {date: $date, id: $id}) MERGE (s)-[r:LOST]->(d)"\n\tparams = {\'name\': args[\'name\'], \'date\': args[\'date\'], \'id\': args[\'id\']}\n\tg.query(q, params)\n')
        self.assertIn(func, funcs)

    # Test creation of ontology graph from schema
    def test_schema_to_graph(self):
        s = Schema()

        f = s.add_entity('Fighter')
        f.add_attribute('name', str, unique=True)
        f.add_attribute('height', int, desc="Fighter's height in cm")

        m = s.add_entity('Match')
        m.add_attribute('date', int, unique=True, desc="Match date in UNIX timestamp")
        m.add_attribute('id', int, unique=True, desc="Fight number")
        m.add_attribute('rounds', int, desc="Number of rounds, either 3 or 5")

        s.add_relation('WON', f, m)
        s.add_relation('LOST', f, m)

        db = FalkorDB()
        g = db.select_graph('UFC')
        s.save_graph(g)

        # Assert topology graph

        # Expecting 2 nodes and 2 edges
        res = g.query("MATCH (n) RETURN count(n)").result_set[0][0]
        self.assertEqual(res, 2)

        res = g.query("MATCH ()-[e]->() RETURN count(e)").result_set[0][0]
        self.assertEqual(res, 2)

        # Validate Fighter ndoe
        n = g.query("MATCH (f:Fighter) RETURN f").result_set[0][0]

        self.assertEqual(n.labels[0], "Fighter")

        self.assertEqual(n.properties['name'][0], 'str')
        self.assertEqual(n.properties['name'][1], "Fighter's name")
        self.assertEqual(n.properties['name'][2], True)
        self.assertEqual(n.properties['name'][3], False)

        self.assertEqual(n.properties['height'][0], 'int')
        self.assertEqual(n.properties['height'][1], "Fighter's height in cm")
        self.assertEqual(n.properties['height'][2], False)
        self.assertEqual(n.properties['height'][3], False)

        # Validate Match ndoe
        n = g.query("MATCH (m:Match) RETURN m").result_set[0][0]

        self.assertEqual(n.labels[0], "Match")

        self.assertEqual(n.properties['rounds'][0], 'int')
        self.assertEqual(n.properties['rounds'][1], "Number of rounds, either 3 or 5")
        self.assertEqual(n.properties['rounds'][2], False)
        self.assertEqual(n.properties['rounds'][3], False)

        self.assertEqual(n.properties['date'][0], 'int')
        self.assertEqual(n.properties['date'][1], "Match date in UNIX timestamp")
        self.assertEqual(n.properties['date'][2], True)
        self.assertEqual(n.properties['date'][3], False)

        self.assertEqual(n.properties['id'][0], 'int')
        self.assertEqual(n.properties['id'][1], "Fight number")
        self.assertEqual(n.properties['id'][2], True)
        self.assertEqual(n.properties['id'][3], False)

        # Validate WON edge
        res = g.query("MATCH (src)-[:WON]->(dest) RETURN src, dest").result_set[0]
        src  = res[0]
        dest = res[1]

        self.assertEqual(src.labels[0], "Fighter")
        self.assertEqual(dest.labels[0], "Match")

        # Validate LOST edge
        res = g.query("MATCH (src)-[:LOST]->(dest) RETURN src, dest").result_set[0]
        src  = res[0]
        dest = res[1]

        self.assertEqual(src.labels[0], "Fighter")
        self.assertEqual(dest.labels[0], "Match")

    # Test creation of schema from ontology graph
    def test_schema_from_graph(self):
        # Create ontology graph from schema
        s = Schema()

        f = s.add_entity('Fighter')
        f.add_attribute('name', str, unique=True)
        f.add_attribute('height', int, desc="Fighter's height in cm")

        m = s.add_entity('Match')
        m.add_attribute('date', int, unique=True, desc="Match date in UNIX timestamp")
        m.add_attribute('id', int, unique=True, desc="Fight number")
        m.add_attribute('rounds', int, desc="Number of rounds, either 3 or 5")

        s.add_relation('WON', f, m)
        s.add_relation('LOST', f, m)

        db = FalkorDB()
        g = db.select_graph('UFC')
        s.save_graph(g)

        # Recreate schema from ontology graph
        self.assertEqual(s, Schema.from_graph(g))

if __name__ == '__main__':
    unittest.main()
