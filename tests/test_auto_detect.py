import unittest
from graphrag_sdk import Source
from graphrag_sdk.schema import Schema

class TestSchemaAutoDetect(unittest.TestCase):
    def test_schema(self):
        sources = [Source("./data/madoff.txt")]
        s = Schema.auto_detect(sources)
        print(f"schema: {s.to_JSON()}")

if __name__ == '__main__':
    unittest.main()
