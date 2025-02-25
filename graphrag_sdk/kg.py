import logging
import warnings
from falkordb import FalkorDB
from typing import Optional, Union
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.chat_session import ChatSession
from graphrag_sdk.attribute import AttributeType, Attribute
from graphrag_sdk.helpers import map_dict_to_cypher_properties
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.steps.extract_data_step import ExtractDataStep
from graphrag_sdk.fixtures.prompts import (GRAPH_QA_SYSTEM, CYPHER_GEN_SYSTEM,
                                CYPHER_GEN_PROMPT, GRAPH_QA_PROMPT, CYPHER_GEN_PROMPT_WITH_HISTORY)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class KnowledgeGraph:
    """Knowledge Graph model data as a network of entities and relations
    To create one it is best to provide a ontology which will define the graph's ontology
    In addition to a set of sources from which entities and relations will be extracted.
    """

    def __init__(
        self,
        name: str,
        model_config: KnowledgeGraphModelConfig,
        ontology: Optional[Ontology] = None,
        host: Optional[str] = "127.0.0.1",
        port: Optional[int] = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cypher_system_instruction: Optional[str] = None,
        qa_system_instruction: Optional[str] = None,
        cypher_gen_prompt: Optional[str] = None,
        qa_prompt: Optional[str] = None,
        cypher_gen_prompt_history: Optional[str] = None,
    ):
        """
        Initialize Knowledge Graph

        Args:
            name (str): Knowledge graph name.
            model (GenerativeModel): The Google GenerativeModel to use.
            host (str): FalkorDB hostname.
            port (int): FalkorDB port number.
            username (Optional[str]): FalkorDB username.
            password (Optional[str]): FalkorDB password.
            ontology (Optional[str]): Ontology to use.
            cypher_system_instruction (Optional[str]): Cypher system instruction. Make sure you have {ontology} in the instruction.
            qa_system_instruction (Optional[str]): QA system instruction.
            cypher_gen_prompt (Optional[str]): Cypher generation prompt. Make sure you have {question} in the prompt.
            qa_prompt (Optional[str]): QA prompt. Make sure you have {question}, {context} and {cypher} in the prompt.
            cypher_gen_prompt_history (Optional[str]): Cypher generation prompt with history. Make sure you have {question} and {last_answer} in the prompt.
        """

        if not isinstance(name, str) or name == "":
            raise Exception("name should be a non empty string")

        # Connect to database
        self.db = FalkorDB(host=host, port=port, username=username, password=password)
        self.graph = self.db.select_graph(name)
        ontology_graph = self.db.select_graph("{" + name + "}" + "_schema")

        # Load / Save ontology to database
        if ontology is None:
            # Load ontology from DB
            ontology = Ontology.from_schema_graph(ontology_graph)
            
            if len(ontology.entities) == 0:
                raise Exception("The ontology is empty. Load a valid ontology or create one using the ontology module.")
        else:
            # Save ontology to DB
            ontology.save_to_graph(ontology_graph)

        self._ontology = ontology
        self._name = name
        self._model_config = model_config
        self.failed_documents = set([])

        if cypher_system_instruction is None:
            cypher_system_instruction = CYPHER_GEN_SYSTEM
        else:
            if "{ontology}" not in cypher_system_instruction:
                warnings.warn("Cypher system instruction should contain {ontology}", category=UserWarning)

        if qa_system_instruction is None:
            qa_system_instruction = GRAPH_QA_SYSTEM

        if cypher_gen_prompt is None:
            cypher_gen_prompt = CYPHER_GEN_PROMPT
        else:
            if "{question}" not in cypher_gen_prompt:
                raise Exception("Cypher generation prompt should contain {question}")

        if qa_prompt is None:
            qa_prompt = GRAPH_QA_PROMPT
        else:
            if "{question}" not in qa_prompt or "{context}" not in qa_prompt:
                raise Exception("QA prompt should contain {question} and {context}")
            if "{cypher}" not in qa_prompt:
                warnings.warn("QA prompt should contain {cypher}", category=UserWarning)

        if cypher_gen_prompt_history is None:
            cypher_gen_prompt_history = CYPHER_GEN_PROMPT_WITH_HISTORY
        else:
            if "{question}" not in cypher_gen_prompt_history:
                raise Exception("Cypher generation prompt with history should contain {question}")
            if "{last_answer}" not in cypher_gen_prompt_history:
                warnings.warn("Cypher generation prompt with history should contain {last_answer}", category=UserWarning)

        # Assign the validated values
        self.cypher_system_instruction = cypher_system_instruction
        self.qa_system_instruction = qa_system_instruction
        self.cypher_gen_prompt = cypher_gen_prompt
        self.qa_prompt = qa_prompt
        self.cypher_gen_prompt_history = cypher_gen_prompt_history

    # Attributes

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise AttributeError("Cannot modify the 'name' attribute")

    @property
    def ontology(self):
        return self._ontology

    @ontology.setter
    def ontology(self, value):
        self._ontology = value

    def list_sources(self) -> list[AbstractSource]:
        """
        List of sources associated with knowledge graph

        Returns:
            list[AbstractSource]: sources
        """

        return [s.source for s in self.sources]

    def process_sources(
        self, sources: list[AbstractSource], instructions: Optional[str] = None, hide_progress: Optional[bool] = False
    ) -> None:
        """
        Add entities and relations found in sources into the knowledge-graph

        Args:
            sources (list[AbstractSource]): list of sources to extract knowledge from
            instructions (Optional[str]): Instructions for processing.
            hide_progress (Optional[bool]): hide progress bar
        """

        if self.ontology is None:
            raise Exception("Ontology is not defined")

        # Create graph with sources
        self._create_graph_with_sources(sources, instructions, hide_progress)


    def _create_graph_with_sources(
        self, sources: Optional[list[AbstractSource]] = None, instructions: Optional[str] = None, hide_progress: Optional[bool] = False
    ) -> None:
        """
        Create a graph using the provided sources.
        
        Args:
            sources (Optional[list[AbstractSource]]): List of sources.
            instructions (Optional[str]): Instructions for the graph creation.
        """
        step = ExtractDataStep(
            sources=list(sources),
            ontology=self.ontology,
            model=self._model_config.extract_data,
            graph=self.graph,
            hide_progress=hide_progress,
        )

        self.failed_documents = step.run(instructions)
                
    def delete(self) -> None:
        """
        Deletes the knowledge graph and any other related resource
        e.g. Ontology, data graphs
        """
        # List available graphs
        available_graphs = self.db.list_graphs()

        # Delete KnowledgeGraph
        if self.name in available_graphs:
            self.graph.delete()

        # Nullify all attributes
        for key in self.__dict__.keys():
            setattr(self, key, None)

    def chat_session(self) -> ChatSession:
        """
        Create a new chat session.
        
        Returns:
            ChatSession: A new chat session instance.
        """
        chat_session = ChatSession(self._model_config, self.ontology, self.graph, self.cypher_system_instruction,
                                   self.qa_system_instruction, self.cypher_gen_prompt, self.qa_prompt, self.cypher_gen_prompt_history)
        return chat_session
    def add_node(self, entity: str, attributes: dict) -> None:
        """
        Add a node to the knowledge graph, checking if it matches the ontology

        Args:
            label (str): label of the node
            attributes (dict): node attributes
        """

        self._validate_entity(entity, attributes)

        # Add node to graph
        self.graph.query(
            f"MERGE (n:{entity} {map_dict_to_cypher_properties(attributes)})"
        )

    def add_edge(
        self,
        relation: str,
        source: str,
        target: str,
        source_attr: Optional[dict] = None,
        target_attr: Optional[dict] = None,
        attributes: Optional[dict] = None,
    ) -> None:
        """
        Add an edge to the knowledge graph, checking if it matches the ontology

        Args:
            relation (str): relation label
            source (str): source entity label
            target (str): target entity label
            source_attr (Optional[dict]): Source entity attributes.
            target_attr (Optional[dict]): Target entity attributes.
            attributes (Optional[dict]): Relation attributes.
        """

        source_attr = source_attr or {}
        target_attr = target_attr or {}
        attributes = attributes or {}

        self._validate_relation(
            relation, source, target, source_attr, target_attr, attributes
        )

        # Add relation to graph
        self.graph.query(
            f"MATCH (s:{source} {map_dict_to_cypher_properties(source_attr)}) MATCH (t:{target} {map_dict_to_cypher_properties(target_attr)}) MERGE (s)-[r:{relation} {map_dict_to_cypher_properties(attributes)}]->(t)"
        )

    def _validate_entity(self, entity: str, attributes: str) -> None:
        """
        Validate if the entity exists in the ontology and check its attributes.
        
        Args:
            entity (str): Entity label.
            attributes (dict): Entity attributes.
        """
        ontology_entity = self.ontology.get_entity_with_label(entity)

        if ontology_entity is None:
            raise Exception(f"Entity {entity} not found in ontology")

        self._validate_attributes_dict(attributes, ontology_entity.attributes)

    def _validate_relation(
        self,
        relation: str,
        source: str,
        target: str,
        source_attr: dict,
        target_attr: dict,
        attributes: dict,
    ) -> None:
        """
        Validate if the relation exists in the ontology and check its attributes.
        
        Args:
            relation (str): Relation label.
            source (str): Source entity label.
            target (str): Target entity label.
            source_attr (dict): Source entity attributes.
            target_attr (dict): Target entity attributes.
            attributes (dict): Relation attributes.
        """
        ontology_relations = self.ontology.get_relations_with_label(relation)

        found_relation = [
            relation
            for relation in ontology_relations
            if relation.source.label == source and relation.target.label == target
        ]
        if len(ontology_relations) == 0 or len(found_relation) == 0:
            raise Exception(f"Relation {relation} not found in ontology")

        self._validate_attributes_dict(attributes, found_relation[0].attributes)

        self._validate_entity(source, source_attr)
        self._validate_entity(target, target_attr)

    def _validate_attributes_dict(
        self, attr_dict: dict, attributes_list: list[Attribute]
    ):
        # validate attributes
        for attr in attributes_list:
            if attr.name not in attr_dict:
                if attr.required or attr.unique:
                    raise Exception(f"Attribute {attr.name} is required")

        for attr in attr_dict.keys():
            valid_attr = [a for a in attributes_list if a.name == attr]
            if len(valid_attr) == 0:
                raise Exception(f"Invalid attribute {attr}")
            valid_attr = valid_attr[0]

            if valid_attr.type == AttributeType.STRING:
                if not isinstance(attr_dict[attr], str):
                    raise Exception(f"Attribute {attr} should be a string")
            elif valid_attr.type == AttributeType.NUMBER:
                if not isinstance(attr_dict[attr], int) and not isinstance(
                    attr_dict[attr], float
                ):
                    raise Exception(f"Attribute {attr} should be an number")
            elif valid_attr.type == AttributeType.BOOLEAN:
                if not isinstance(attr_dict[attr], bool):
                    raise Exception(f"Attribute {attr} should be a boolean")
