import unittest
from rag_sdk.kg import KnowledgeGraph
from rag_sdk.Schema import Schema

class TestKG(unittest.TestCase):
    def test_kg_sources(self):
        g = KnowledgeGraph("UFC")

        data_0 = g.add_source("data_0.pdf")
        data_1 = g.add_source("data_1.PDF")
        data_2 = g.add_source("data_2.txt")
        data_3 = g.add_source("data_3")

        sources = g.list_sources()
        self.assertEqual(len(sources), 4)

        g.remove_source("data_0.pdf")
        g.remove_source("data_0.pdf")

        sources = g.list_sources()
        self.assertEqual(len(sources), 3)

        g.remove_source(data_2)
        g.remove_source("data_1.PDF")
        g.remove_source("data_3")

        sources = g.list_sources()
        self.assertEqual(len(sources), 0)

    def test_kg_creation(self):
        # Create schema
        s = Schema()
        s.add_entity('Actor').add_attribute('name', str, unique=True)
        s.add_entity('Movie').add_attribute('title', str, unique=True)
        s.add_relation("ACTED", 'Actor', 'Movie')

        g = KnowledgeGraph("IMDB", schema=s)
        g.add_source("./data/madoff.txt")

        g.create()

        answer, messages = g.ask("List a few actors")
        print(f"answer: {answer}")

        answer, messages = g.ask("list additional actors", messages)
        print(f"answer: {answer}")

if __name__ == '__main__':
    unittest.main()
