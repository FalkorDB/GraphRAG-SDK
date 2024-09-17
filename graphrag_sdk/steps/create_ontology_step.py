from graphrag_sdk.steps.Step import Step
from graphrag_sdk.source import AbstractSource
from graphrag_sdk.document import Document
from concurrent.futures import Future, ThreadPoolExecutor, wait
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.fixtures.prompts import (
    CREATE_ONTOLOGY_SYSTEM,
    CREATE_ONTOLOGY_PROMPT,
    FIX_ONTOLOGY_PROMPT,
    FIX_JSON_PROMPT,
    BOUNDARIES_PREFIX,
)
import logging
from graphrag_sdk.helpers import extract_json
from ratelimit import limits, sleep_and_retry
import time
from graphrag_sdk.models import (
    GenerativeModel,
    GenerativeModelChatSession,
    GenerativeModelConfig,
    GenerationResponse,
    FinishReason,
)
import json
from typing import Optional

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
    ) -> None:
        self.sources = sources
        self.ontology = ontology
        self.model = model.with_system_instruction(CREATE_ONTOLOGY_SYSTEM)
        self.config = config

    def _create_chat(self):
        return self.model.start_chat({"response_validation": False})

    def run(self, boundaries: Optional[str] = None):
        tasks: list[Future[Ontology]] = []
        with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            # extract entities and relationships from each page

            documents = [
                document for source in self.sources for document in source.load()
            ]
            for source in documents:
                task = executor.submit(
                    self._process_source,
                    self._create_chat(),
                    source,
                    self.ontology,
                    boundaries,
                )
                tasks.append(task)

            # Wait for all tasks to complete
            wait(tasks)

        for task in tasks:
            self.ontology = self.ontology.merge_with(task.result())

        if len(self.ontology.entities) == 0:
            raise Exception("Failed to create ontology")

        self.ontology = self._fix_ontology(self._create_chat(), self.ontology)

        return self.ontology

    def _process_source(
        self,
        chat_session: GenerativeModelChatSession,
        document: Document,
        o: Ontology,
        boundaries: Optional[str] = None,
    ):
        text = document.content[: self.config["max_input_tokens"]]

        user_message = CREATE_ONTOLOGY_PROMPT.format(
            text = text,
            boundaries = BOUNDARIES_PREFIX.format(user_boundaries=boundaries) if boundaries is not None else "", 
        )

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
                logger.error(f"Failed to fix JSON: {e}  {json_fix_response.text}")
                data = None

        if data is None:
            return o
        
        try:
            new_ontology = Ontology.from_json(data)
        except Exception as e:
            logger.error(f"Exception while extracting JSON: {e}")
            new_ontology = None

        if new_ontology is not None:
            o = o.merge_with(new_ontology)

        logger.debug(f"Processed document: {document}")

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
