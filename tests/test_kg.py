import unittest
from rag_sdk import KnowledgeGraph, Source
from rag_sdk.Schema import Schema

class TestKG(unittest.TestCase):
    def test_kg_creation(self):
        # Create schema
        s = Schema()
        actor = s.add_entity('Actor')
        actor.add_attribute('name', str, unique=True)

        movie = s.add_entity('Movie')
        movie.add_attribute('title', str, unique=True)

        s.add_relation("ACTED", actor, movie)

        g = KnowledgeGraph("IMDB", schema=s)
        g.process_sources([Source("./data/madoff.txt")])

        answer, messages = g.ask("List a few actors")
        print(f"answer: {answer}")

        answer, messages = g.ask("list additional actors", messages)
        print(f"answer: {answer}")

if __name__ == '__main__':
    unittest.main()
