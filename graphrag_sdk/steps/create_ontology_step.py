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
        config: Optional[dict] = None,
        hide_progress: Optional[bool] = False,
    ) -> None:
        """
        Initialize the CreateOntologyStep.
        
        Args:
            sources (List[AbstractSource]): List of sources from which ontology is created.
            ontology (Ontology): The initial ontology to be merged with new extracted data.
            model (GenerativeModel): The generative model used for processing and creating the ontology.
            config (Optional[dict]): Configuration for the step, including thread workers and token limits. Defaults to standard config.
            hide_progress (bool): Flag to hide the progress bar. Defaults to False.
        """
        self.sources = sources
        self.ontology = ontology
        self.model = model
        if config is None:
            self.config = {
                "max_workers": 16,
                "max_input_tokens": 500000,
                "max_output_tokens": 8192,
            }
        else:
            self.config = config
        self.hide_progress = hide_progress
        self.process_files = 0
        self.counter_lock = Lock()

    def _create_chat(self) -> GenerativeModelChatSession:
        """
        Create a new chat session with the generative model.
        
        Returns:
            GenerativeModelChatSession: A session for interacting with the generative model.
        """
        return self.model.start_chat(CREATE_ONTOLOGY_SYSTEM)

    def run(self, boundaries: Optional[str] = None):
        """
        Execute the ontology creation process by extracting data from all sources.
        
        Args:
            boundaries (Optional[str]): Additional boundaries or constraints for the ontology creation.
            
        Returns:
            Ontology: The final ontology after merging with extracted data.
        
        Raises:
            Exception: If ontology creation fails and no entities are found.
        """
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
        retries: Optional[int] = 1,
    ):
        """
        Process a single document and extract ontology data.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the generative model.
            document (Document): The document to extract data from.
            o (Ontology): The current ontology to be merged with extracted data.
            boundaries (Optional[str]): Constraints for data extraction.
            retries (Optional[int]) Number of retries for processing the document.
            
        Returns:
            Ontology: The updated ontology after processing the document.
        """
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
        """
        Fix and validate the ontology using the generative model.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the model.
            o (Ontology): The ontology to fix and validate.
            
        Returns:
            Ontology: The fixed and validated ontology.
        """
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
        retry: Optional[int] = 6,
    ):
        """
        Call the generative model with retries and rate limiting.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the model.
            prompt (str): The prompt to send to the model.
            retry (Optional[int]): Number of retries if quota is exceeded or errors occur.
            
        Returns:
            GenerationResponse: The model's response.
            
        Raises:
            Exception: If the model fails after exhausting retries.
        """
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
