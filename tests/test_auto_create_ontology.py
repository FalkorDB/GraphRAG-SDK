from dotenv import load_dotenv
load_dotenv()
from graphrag_sdk.ontology import Ontology
import unittest
from graphrag_sdk.source import Source
from graphrag_sdk.models.litellm import LiteModel
import os
import logging

logging.basicConfig(level=logging.DEBUG)


class TestAutoDetectOntology(unittest.TestCase):
    """
    Test auto-detect ontology
    """

    def test_auto_detect_ontology(self):

        file_path = "tests/data/madoff.txt"

        sources = [Source(file_path)]

        model = LiteModel(model_name="gemini/gemini-2.0-flash")

        boundaries = """
          Extract entities and relationships from each page
        """
        ontology = Ontology.from_sources(sources, boundaries=boundaries, model=model)

        logging.info(f"Ontology: {ontology.to_json()}")
