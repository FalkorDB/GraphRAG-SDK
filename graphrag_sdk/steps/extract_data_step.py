import os
import time
import json
import logging
from tqdm import tqdm
from uuid import uuid4
from falkordb import Graph
from threading import Lock
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.document import Document
from ratelimit import limits, sleep_and_retry
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.models.model import OutputMethod
from concurrent.futures import Future, ThreadPoolExecutor
from graphrag_sdk.helpers import extract_json, map_dict_to_cypher_properties
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models import (
    GenerativeModel,
    GenerativeModelChatSession,
    GenerationResponse,
    FinishReason,
)
from graphrag_sdk.fixtures.prompts import (
    EXTRACT_DATA_SYSTEM,
    EXTRACT_DATA_PROMPT,
    FIX_JSON_PROMPT,
    COMPLETE_DATA_EXTRACTION,
)

RENDER_STEP_SIZE = 0.5

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ExtractDataStep(Step):
    """
    Extract Data Step
    """

    def __init__(
        self,
        sources: list[AbstractSource],
        ontology: Ontology,
        model: GenerativeModel,
        graph: Graph,
        config: dict = {
            "max_workers": 16,
            "max_input_tokens": 500000,
            "max_output_tokens": 8192,
        },
        hide_progress: bool = False,
    ) -> None:
        self.sources = sources
        self.ontology = ontology
        self.config = config
        self.model = model.with_system_instruction(
            EXTRACT_DATA_SYSTEM.replace("#ONTOLOGY", str(self.ontology.to_json()))
        )
        self.graph = graph
        self.hide_progress = hide_progress
        self.process_files = 0
        self.counter_lock = Lock()
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def _create_chat(self):
        return self.model.start_chat({"response_validation": False})

    def run(self, instructions: str = None) -> list[AbstractSource]:

        tasks: list[Future[Ontology]] = []
        
        with tqdm(total=len(self.sources), desc="Process Documents", disable=self.hide_progress) as pbar:
            with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:

                # Process each source document in parallel
                for source in self.sources:
                    task_id = "extract_data_step_" + str(uuid4())
                    task = executor.submit(
                        self._process_source,
                        task_id,
                        self._create_chat(),
                        source,
                        self.ontology,
                        self.graph,
                        instructions,
                    )
                    tasks.append(task)
                    
                # Wait for all tasks to be completed
                while any(task.running() or not task.done() for task in tasks):
                    time.sleep(RENDER_STEP_SIZE)
                    with self.counter_lock:
                        pbar.n = self.process_files
                    pbar.refresh()

        failed_sources = [self.sources[idx] for idx, task in enumerate(tasks) if task.exception()]     
        return failed_sources

    def _process_source(
        self,
        task_id: str,
        chat_session: GenerativeModelChatSession,
        source: AbstractSource,
        ontology: Ontology,
        graph: Graph,
        source_instructions: str = "",
        instructions: str = "",
        retries: int = 1,
    ):
        try:
            _task_logger = logging.getLogger(task_id)
            _task_logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(f"logs/{task_id}.log")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
            )
            fh.setLevel(logging.DEBUG)

            _task_logger.addHandler(fh)

            logger.debug(f"Processing task: {task_id}")
            _task_logger.debug(f"Processing task: {task_id}")
            
            document = next(source.load())
            source_instructions = source.instruction
                
            text = document.content[: self.config["max_input_tokens"]]
            user_message = EXTRACT_DATA_PROMPT.format(
                text=text,
                instructions="\n".join(
                    [
                        source_instructions if source_instructions is not None else "",
                        instructions if instructions is not None else "",
                    ]
                ),
                max_tokens=self.config["max_output_tokens"],
                ontology=str(ontology.to_json()),
            )

            _task_logger.debug("User message: " + user_message.replace("\n", " "))

            responses: list[GenerationResponse] = []
            response_idx = 0

            responses.append(self._call_model(chat_session, user_message, output_method=OutputMethod.JSON))

            _task_logger.debug(f"Model response: {responses[response_idx].text}")

            while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS and response_idx < retries:
                _task_logger.debug("Asking model to continue")
                response_idx += 1
                responses.append(self._call_model(chat_session, COMPLETE_DATA_EXTRACTION, output_method=OutputMethod.JSON))
                _task_logger.debug(
                    f"Model response after continue: {responses[response_idx].text}"
                )

            if responses[response_idx].finish_reason != FinishReason.STOP:
                _task_logger.debug(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )
                raise Exception(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )

            # Full json response is in the last response
            last_respond = responses[-1].text

            try:
                data = json.loads(extract_json(last_respond))
            except Exception as e:
                _task_logger.debug(f"Error extracting JSON: {e}")
                _task_logger.debug(f"Prompting model to fix JSON")
                json_fix_response = self._call_model(
                    self._create_chat(),
                    FIX_JSON_PROMPT.format(json=last_respond, error=str(e)),
                    output_method=OutputMethod.JSON,
                )
                data = json.loads(extract_json(json_fix_response.text))
                _task_logger.debug(f"Fixed JSON: {data}")

            if "entities" not in data or "relations" not in data:
                _task_logger.debug(
                    f"Invalid data format. Missing entities or relations. {data}"
                )
                raise Exception(
                    f"Invalid data format. Missing 'entities' or 'relations' in JSON."
                )
            for entity in data["entities"]:
                try:
                    self._create_entity(graph, entity, ontology)
                except Exception as e:
                    _task_logger.error(f"Error creating entity: {e}")
                    continue

            for relation in data["relations"]:
                try:
                    self._create_relation(graph, relation, ontology)
                except Exception as e:
                    _task_logger.error(f"Error creating relation: {e}")
                    continue
            
        except Exception as e:
            logger.exception(f"Task id: {task_id} failed - {e}")
            raise e
        finally:
            with self.counter_lock:
                self.process_files += 1

    def _create_entity(self, graph: Graph, args: dict, ontology: Ontology):
        # Get unique attributes from entity
        entity = ontology.get_entity_with_label(args["label"])
        if entity is None:
            print(f"Entity with label {args['label']} not found in ontology")
            return None
        unique_attributes_schema = [attr for attr in entity.attributes if attr.unique]
        unique_attributes = {
            attr.name: (
                args["attributes"][attr.name] if attr.name in args["attributes"] else ""
            )
            for attr in unique_attributes_schema
        }
        unique_attributes_text = map_dict_to_cypher_properties(unique_attributes)
        non_unique_attributes = {
            attr.name: args["attributes"][attr.name]
            for attr in entity.attributes
            if not attr.unique and attr.name in args["attributes"]
        }
        non_unique_attributes_text = map_dict_to_cypher_properties(
            non_unique_attributes
        )
        set_statement = (
            f"SET n += {non_unique_attributes_text}"
            if len(non_unique_attributes.keys()) > 0
            else ""
        )
        query = f"MERGE (n:{args['label']} {unique_attributes_text}) {set_statement}"
        logger.debug(f"Query: {query}")
        result = graph.query(query)
        return result

    def _create_relation(self, graph: Graph, args: dict, ontology: Ontology):
        relations = ontology.get_relations_with_label(args["label"])
        if len(relations) == 0:
            print(f"Relations with label {args['label']} not found in ontology")
            return None
        source_unique_attributes = (
            args["source"]["attributes"]
            if "source" in args and "attributes" in args["source"]
            else {}
        )
        source_unique_attributes_text = map_dict_to_cypher_properties(
            source_unique_attributes
        )

        target_unique_attributes = (
            args["target"]["attributes"]
            if "target" in args and "attributes" in args["target"]
            else {}
        )
        target_unique_attributes_text = map_dict_to_cypher_properties(
            target_unique_attributes
        )

        relation_attributes = (
            map_dict_to_cypher_properties(args["attributes"])
            if "attributes" in args
            else {}
        )
        set_statement = (
            f"SET r += {relation_attributes}"
            if "attributes" in args
            and len(
                args["attributes"]
                if isinstance(args["attributes"], list)
                else args["attributes"].keys()
            )
            > 0
            else ""
        )
        query = f"MATCH (s:{args['source']['label']} {source_unique_attributes_text}) MATCH (d:{args['target']['label']} {target_unique_attributes_text}) MERGE (s)-[r:{args['label']}]->(d) {set_statement}"
        logger.debug(f"Query: {query}")
        result = graph.query(query)
        return result

    @sleep_and_retry
    @limits(calls=15, period=60)
    def _call_model(
        self,
        chat_session: GenerativeModelChatSession,
        prompt: str,
        retry: int = 6,
        output_method: OutputMethod = OutputMethod.DEFAULT
    ):
        try:
            return chat_session.send_message(prompt, output_method=output_method)
        except Exception as e:
            # If exception is caused by quota exceeded, wait 10 seconds and try again for 6 times
            if "Quota exceeded" in str(e) and retry > 0:
                time.sleep(10)
                retry -= 1
                return self._call_model(chat_session, prompt, retry)
            else:
                if retry == 0:
                    logger.error("Quota exceeded")
                raise e
