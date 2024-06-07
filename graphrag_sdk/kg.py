import queue
from openai import OpenAI
import concurrent.futures
from typing_extensions import AbstractSet

from .prompts import *
from .template import TEMPLATE
from openai import OpenAI
from falkordb import FalkorDB
from .query_graph import query
from .schema.schema import Schema
from .source import Source, AbstractSource
from .schema.tools_gen import schema_to_tools
from .schema.functions_gen import schema_to_functions

import logging
logger = logging.getLogger(__name__)

class KnowledgeGraph(object):
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

        if not isinstance(name, str) or name == "":
            raise Exception("name should be a non empty string")

        # connect to database
        self.db = FalkorDB(host=host, port=port, username=username, password=password)
        self.graph = self.db.select_graph(name)

        self.model = model
        self._name = name
        self._schema = schema
        self.client = None # OpenAI client
        self.sources = set([])
        self.ontolegy_ratio = 0.1 # Sampaling ratio for ontolegy detection

        # in case schema is None
        # try to load schema from FalkorDB
        if schema is None:
            schema_name = self._schema_name()
            if schema_name in self.db.list_graphs():
                schema_graph = self.db.select_graph(schema_name)
                self._schema = Schema.from_graph(schema_graph)

                logger.info(f"Schema detected")
                logger.debug(f"Schema: {self.schema.to_JSON()}")

    # Attributes

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise AttributeError("Cannot modify the 'name' attribute")

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schame(self, value):
        raise AttributeError("Cannot modify the 'schema' attribute")

    def list_sources(self) -> list[AbstractSource]:
        """
        List of sources associated with knowledge graph

        Returns:
            list[AbstractSource]: sources
        """

        return [s.source for s in self.sources]

    def process_sources(self, sources: list[AbstractSource]) -> None:
        """
        Add entities and relations found in sources into the knowledge-graph

        Parameters:
            sources (list[AbstractSource]): list of sources to extract knowledge from
        """

        # Make sure knowledge graph is created.
        if not 'build_graph_from_sources' in globals() or self.client is None:
            self._create()

        # Run assistant
        result = build_graph_from_sources(self, self.client, sources)

        # Add processed sources
        for src in sources:
            self.sources.add(src)

    def _schema_name(self) -> str:
        """
        Generate a name for the schema based on the Knowledge Graph name
        """

        return f"{self.name}_schema"

    def _createKGWithSchema(self):
        # Save schema as an ontology graph
        schema_graph = self.db.select_graph(self._schema_name())
        self.schema.save_graph(schema_graph)

        # Create Knowledge Graph construction assistant

        # Generate assistant's toolings
        tools = schema_to_tools(self.schema)

        # Generate functions from schema
        funcs = schema_to_functions(self.schema)
        functions_names = [func[0] for func in funcs]
        functions_impl  = [func[1] for func in funcs]

        # Generate assistant
        # Generate code from template
        code = TEMPLATE
        code = code.replace("<TOOLS>", ",\n".join([str(tool) for tool in tools]))
        code = code.replace("<GRAPH_ID>", "\"" + self.name + "\"")
        code = code.replace("<FUNCTIONS>", "\n".join(functions_impl))
        code = code.replace("<AVAILABLE_FUNCTIONS>", '{' + ",\n".join([f"'{f}': {f}" for f in functions_names]) +'}')

        # Execute the code string in the global namespace
        # Introduces the 'build_graph_from_sources' function
        # TODO: note multiple calls to _createKGWithSchema will overide one another.
        exec(code, globals())

    def _create(self) -> None:
        """
        Create knowledge graph
        Extract entities and relations from sources
        """

        if self.schema is None:
            raise Exception("Can not create knowledge graph, schema missing")

        # Create OpenAI client
        if self.client is None:
            self.client = OpenAI()

        self._createKGWithSchema()

    def ask(self, question:str, history:list|None=None) -> tuple[str, list]:
        """
        Query the knowledge graph using natural language
        if the query is asked as part of a longer conversation make sure to
        include past history.

        Rreturns:
            tuple[str, list]: answer, conversation history

         Example:
            >>> ans, history = kg.ask("Which actor has the most oscars")
            >>> ans, history = kg.ask("List a few movies in which that actored played in", history)
        """

        if history is None:
            history = []

        answer = query(self, question, history, self.model)
        return (answer, history)

    def delete(self) -> None:
        """
        Deletes the knowledge graph and any other related resource
        e.g. Schema, OpenAI assistant
        """

        # Delete OpenAI KnowledgeGraph creation assistant
        client = OpenAI()
        assistant_ids = []
        for assistant in client.beta.assistants.list():
            if assistant.name == self.name:
                # Do not break, there might be multiple assistants with the same name.
                assistant_ids.append(assistant.id)

        for assistant_id in assistant_ids:
            # Delete assistant
            client.beta.assistants.delete(assistant_id)

        # List available graphs
        available_graphs = self.db.list_graphs()

        # Delete KnowledgeGraph
        if self.name in available_graphs:
            self.graph.delete()

        # Delete schema graph
        if self._schema_name() in available_graphs:
            schema_graph = self.db.select_graph(self._schema_name())
            schema_graph.delete()

        # Nullify all attributes
        for key in self.__dict__.keys():
            setattr(self, key, None)
