import unittest
from openai import OpenAI
from falkordb import FalkorDB
from graphrag_sdk import KnowledgeGraph, Source
from graphrag_sdk.schema import Schema

class TestKG(unittest.TestCase):
    def test_kg_creation(self):
        # Create schema
        s = Schema()
        actor = s.add_entity('Actor').add_attribute('name', str, unique=True)
        movie = s.add_entity('Movie').add_attribute('title', str, unique=True)
        s.add_relation("ACTED", actor, movie)

        g = KnowledgeGraph("IMDB", schema=s)
        g.process_sources([Source("./data/madoff.txt")])

        answer, messages = g.ask("List a few actors")
        print(f"answer: {answer}")

        answer, messages = g.ask("list additional actors", messages)
        print(f"answer: {answer}")

    def test_kg_delete(self):
        s = Schema()
        actor = s.add_entity('Actor').add_attribute('name', str, unique=True)
        movie = s.add_entity('Movie').add_attribute('title', str, unique=True)
        s.add_relation("ACTED", actor, movie)

        g = KnowledgeGraph("IMDB", schema=s)
        g.process_sources([Source("./data/madoff.txt")])

        g.delete()

        # Check that:
        # 1. KnowledgeGraph has been removed from FalkorDB.
        # 2. SchemaGraph has been removed from FalkorDB.
        # 3. OpenAI assistant has been deleted.

        db = FalkorDB()
        graphs = db.list_graphs()
        self.assertNotIn("IMDB", graphs)
        self.assertNotIn("IMDB_schema", graphs)

        client = OpenAI()
        assistant_id = None
        for assistant in client.beta.assistants.list():
            self.assertNotEqual(assistant.name, "IMDB")

if __name__ == '__main__':
    unittest.main()
