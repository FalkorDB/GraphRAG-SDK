import time
import json
import logging
from tqdm import tqdm
from threading import Lock
from typing import Optional
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.document import Document
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.helpers import extract_json
from ratelimit import limits, sleep_and_retry
from graphrag_sdk.source import AbstractSource
from concurrent.futures import Future, ThreadPoolExecutor
from graphrag_sdk.models import (
    GenerativeModel,
    GenerativeModelChatSession,
    GenerationResponse,
    FinishReason,
)
from graphrag_sdk.fixtures.prompts import (
    CREATE_ONTOLOGY_SYSTEM,
    CREATE_ONTOLOGY_PROMPT,
    FIX_ONTOLOGY_PROMPT,
    FIX_JSON_PROMPT,
    BOUNDARIES_PREFIX,
)
RENDER_STEP_SIZE = 0.5


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CreateOntologyStep(Step):
    """
    Create Ontology Step
    """

    def __init__(
        self,
        sources: list[AbstractSource],
        ontology: Ontology,
        model: GenerativeModel,
        config: dict = {
            "max_workers": 16,
            "max_input_tokens": 500000,
            "max_output_tokens": 8192,
        },
        hide_progress: bool = False,
    ) -> None:
        self.sources = sources
        self.ontology = ontology
        self.model = model.with_system_instruction(CREATE_ONTOLOGY_SYSTEM)
        self.config = config
        self.hide_progress = hide_progress
        self.process_files = 0
        self.counter_lock = Lock()

    def _create_chat(self):
        return self.model.start_chat({"response_validation": False})

    def run(self, boundaries: Optional[str] = None):
        tasks: list[Future[Ontology]] = []

        with tqdm(total=len(self.sources) + 1, desc="Process Documents", disable=self.hide_progress) as pbar:
            with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:

                # Process each source document in parallel
                for source in self.sources:
                    task = executor.submit(
                        self._process_source,
                        self._create_chat(),
                        source,
                        self.ontology,
                        boundaries,
                    )
                    tasks.append(task)

                # Wait for all tasks to be completed
                while any(task.running() or not task.done() for task in tasks):
                    time.sleep(RENDER_STEP_SIZE)
                    with self.counter_lock:
                        pbar.n = self.process_files
                    pbar.refresh()
                
                # Validate the ontology
                if len(self.ontology.entities) == 0:
                    raise Exception("Failed to create ontology")
                
                # Finalize the ontology
                task_fin = executor.submit(self._fix_ontology, self._create_chat(), self.ontology)
                
                # Wait for the final task to be completed
                while not task_fin.done():
                    time.sleep(RENDER_STEP_SIZE)
                    pbar.refresh()
                pbar.update(1)

        return self.ontology

    def _process_source(
        self,
        chat_session: GenerativeModelChatSession,
        source: AbstractSource,
        o: Ontology,
        boundaries: Optional[str] = None,
        retries: int = 1,
    ):
        try:
            document = next(source.load())
            
            text = document.content[: self.config["max_input_tokens"]]

            user_message = CREATE_ONTOLOGY_PROMPT.format(
                text = text,
                boundaries = BOUNDARIES_PREFIX.format(user_boundaries=boundaries) if boundaries is not None else "", 
            )

            responses: list[GenerationResponse] = []
            response_idx = 0

            responses.append(self._call_model(chat_session, user_message))

            logger.debug(f"Model response: {responses[response_idx]}")

            while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS and response_idx < retries:
                response_idx += 1
                responses.append(self._call_model(chat_session, "continue"))

            if responses[response_idx].finish_reason != FinishReason.STOP:
                raise Exception(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )

            combined_text = " ".join([r.text for r in responses])

            try:
                data = json.loads(extract_json(combined_text))
            except json.decoder.JSONDecodeError as e:
                logger.debug(f"Error extracting JSON: {e}")
                logger.debug(f"Prompting model to fix JSON")
                json_fix_response = self._call_model(
                    self._create_chat(),
                    FIX_JSON_PROMPT.format(json=combined_text, error=str(e)),
                )
                try:
                    data = json.loads(extract_json(json_fix_response.text))
                    logger.debug(f"Fixed JSON: {data}")
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Failed to fix JSON: {e}  {json_fix_response.text}")
                    data = None

            if data is not None:
                try:
                    new_ontology = Ontology.from_json(data)
                except Exception as e:
                    logger.error(f"Exception while extracting JSON: {e}")
                    new_ontology = None

                if new_ontology is not None:
                    o = o.merge_with(new_ontology)

                logger.debug(f"Processed document: {document}")
        except Exception as e:
            logger.exception(f"Failed - {e}")
            raise e
        finally:
            with self.counter_lock:
                self.process_files += 1
            return o

    def _fix_ontology(self, chat_session: GenerativeModelChatSession, o: Ontology):
        logger.debug(f"Fixing ontology...")

        user_message = FIX_ONTOLOGY_PROMPT.format(ontology=o)

        responses: list[GenerationResponse] = []
        response_idx = 0

        responses.append(self._call_model(chat_session, user_message))

        logger.debug(f"Model response: {responses[response_idx]}")

        while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS:
            response_idx += 1
            responses.append(self._call_model(chat_session, "continue"))

        if responses[response_idx].finish_reason != FinishReason.STOP:
            raise Exception(
                f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
            )

        combined_text = " ".join([r.text for r in responses])

        try:
            data = json.loads(extract_json(combined_text))
        except json.decoder.JSONDecodeError as e:
            logger.debug(f"Error extracting JSON: {e}")
            logger.debug(f"Prompting model to fix JSON")
            json_fix_response = self._call_model(
                self._create_chat(),
                FIX_JSON_PROMPT.format(json=combined_text, error=str(e)),
            )
            try:
                data = json.loads(extract_json(json_fix_response.text))
                logger.debug(f"Fixed JSON: {data}")
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Failed to fix JSON: {e} {json_fix_response.text}")
                data = None

        if data is None:
            return o
        
        try:
            new_ontology = Ontology.from_json(data)
        except Exception as e:
            logger.debug(f"Exception while extracting JSON: {e}")
            new_ontology = None

        if new_ontology is not None:
            o = o.merge_with(new_ontology)

        logger.debug(f"Fixed ontology: {o}")

        return o

    @sleep_and_retry
    @limits(calls=15, period=60)
    def _call_model(
        self,
        chat_session: GenerativeModelChatSession,
        prompt: str,
        retry=6,
    ):
        try:
            return chat_session.send_message(prompt)
        except Exception as e:
            # If exception is caused by quota exceeded, wait 10 seconds and try again for 6 times
            if "Quota exceeded" in str(e) and retry > 0:
                time.sleep(10)
                retry -= 1
                return self._call_model(chat_session, prompt, retry)
            else:
                raise e
