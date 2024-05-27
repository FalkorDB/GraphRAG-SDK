import os
import queue
import concurrent.futures
from typing_extensions import AbstractSet

from .prompts import *
from openai import OpenAI
from falkordb import FalkorDB
from .query_graph import query
from .Schema.schema import Schema
from .source import Source, AbstractSource
from .Schema.tools_gen import schema_to_tools
from .Schema.functions_gen import schema_to_functions

class KG(object):
    """Knowledge Graph model data as a network of entities and relations
    To create one it is best to provide a schema which will define the graph's ontology
    In addition to a set of sources from which entities and relations will be extracted.
    """
    def __init__(self, name:str, host:str="127.0.0.1", port:int=6379, username:str|None=None, password:str|None=None,
        model:str="gpt-4-1106-preview", schema:Schema|None=None):
        """
        Initialize Knowledge Graph

        Parameters:
            name (str): Knowledge graph name.
            host (str): FalkorDB hostname.
            port (int): FalkorDB port number.
            username (str|None): FalkorDB username.
            password (str|None): FalkorDB password.
            model (str): OpenAI model to use.
        """

        # connect to database
        self.db = FalkorDB(host=host, port=port, username=username, password=password)
        self.graph = self.db.select_graph(name)

        self.name = name
        self.model = model
        self.schema = schema
        self.client = None # OpenAI client
        self.sources = set([])
        self.ontolegy_ratio = 0.1 # Sampaling ratio for ontolegy detection

    def add_source(self, source:str) -> AbstractSource:
        """
        Add source from which knowledge will be extracted

        Parameters:
            source (src): path to source

        Returns:
            AbstractSource: A source object
        """

        if not isinstance(source, str) or source == "":
            raise Exception("Invalid argument, source should be a none empty string.")

        s = Source(source)
        self.sources.add(s)

        return s

    def remove_source(self, source:str|AbstractSource):
        """
        Remove source from sources

         Parameters:
             source (str|AbstractSource): source to remove
        """

        if not isinstance(source, str) and not isinstance(source, AbstractSource):
            raise Exception("Invalid argument, source should be a string or a Source")

        if isinstance(source, str):
            for s in self.sources:
                if s.source == source:
                    self.sources.remove(s)
                    break
        else:
            self.sources.remove(source)

    def list_sources(self) -> list[AbstractSource]:
        """
        List of sources associated with knowledge graph

        Returns:
            list[AbstractSource]: sources
        """

        return [s.source for s in self.sources]

    def _createKGWithSchema(self):
        # Save schema as an ontology graph
        self.schema.save_graph(self.graph)

        # Create Knowledge Graph construction assistant

        # Generate assistant's toolings
        tools = schema_to_tools(self.schema)

        # Generate functions from schema
        funcs = schema_to_functions(self.schema)
        functions_names = [func[0] for func in funcs]
        functions_impl  = [func[1] for func in funcs]

        # Generate assistant
        # Generate Python from template
        code = None
        with open("./template.py", "r") as f:
            code = f.read()
            code = code.replace("<TOOLS>", ",\n".join([str(tool) for tool in tools]))
            code = code.replace("<GRAPH_ID>", "\"" + self.name + "\"")
            code = code.replace("<FUNCTIONS>", "\n".join(functions_impl))
            code = code.replace("<AVAILABLE_FUNCTIONS>", '{' + ",\n".join([f"'{f}': {f}" for f in functions_names]) +'}')

        # Load template
        import sys
        import importlib.util

        # Write code to a file
        file_path = 'auto_gen.py'
        with open(file_path, 'w') as file:
            file.write(code)

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("auto_gen", file_path)
        dynamic_module = importlib.util.module_from_spec(spec)
        sys.modules["dynamic_module"] = dynamic_module
        spec.loader.exec_module(dynamic_module)

        # Run assistant
        result = dynamic_module.build_graph_from_sources(self, self.client, self.sources)

    def _createKGWithoutSchema(self):
        pass

    def create(self):
        """
        Create knowledge graph
        Extract entities and relations from sources
        """
        self.client = OpenAI()

        if self.schema is not None:
            self._createKGWithSchema()
        else:
            self._createKGWithoutSchema()


    def ask(self, question:str, messages:list=None) -> tuple[str, list]:
        """
        Query the knowledge graph using natural language
        if the query is asked as part of a longer conversation make sure to
        include past messages.

        Rreturns:
            tuple[str, list]: answer, conversation messages

         Example:
            >>> ans, msgs = kg.ask("Which actor has the most oscars")
            >>> ans, msgs = kg.ask("List a few movies in which that actored played in", msgs)
        """

        if messages is None:
            messages = []

        answer = query(self, question, messages, self.model)
        return (answer, messages)
