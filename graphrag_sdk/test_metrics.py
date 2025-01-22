from typing import Optional
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from typing import Optional, List, Union

from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)

from deepeval.metrics.contextual_relevancy.schema import *
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.contextual_recall.template import ContextualRecallTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.contextual_recall.schema import *

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]

class CombineMetrics(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: Optional[str] = "gpt-4o",
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, faithfulness_metric, graph_context_recall_metric, graph_context_relevancy_metric = self.initialize_metrics()
            # Remember, deepeval's default metrics follow the same pattern as your custom metric!
            graph_context_relevancy_metric.measure(test_case)
            graph_context_recall_metric.measure(test_case)
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)

            # Custom logic to set score, reason, and success
            self.set_score_reason_success(relevancy_metric, faithfulness_metric, graph_context_recall_metric, graph_context_relevancy_metric)
            return self.score
        except Exception as e:
            # Set and re-raise error
            self.error = str(e)
            raise


    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Composite Relevancy Faithfulness Metric"


    ######################
    ### Helper methods ###
    ######################
    def initialize_metrics(self):
        graph_context_relevancy_metric = GraphContextualRelevancy(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            strict_mode=self.strict_mode
        )
        
        graph_context_recall_metric = GraphContextualRecall(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            strict_mode=self.strict_mode
        )
        
        relevancy_metric = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        faithfulness_metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        return relevancy_metric, faithfulness_metric, graph_context_recall_metric, graph_context_relevancy_metric

    def set_score_reason_success(
        self,
        relevancy_metric: BaseMetric,
        faithfulness_metric: BaseMetric,
        graph_context_recall_metric: BaseMetric,
        graph_context_relevancy_metric: BaseMetric,
    ):
        # Get scores and reasons for both
        relevancy_score = relevancy_metric.score
        relevancy_reason = relevancy_metric.reason
        faithfulness_score = faithfulness_metric.score
        faithfulness_reason = faithfulness_metric.reason
        graph_context_recall_metric_score = graph_context_recall_metric.score
        graph_context_recall_metric_reason = graph_context_recall_metric.reason
        graph_context_relevancy_metric_score = graph_context_relevancy_metric.score
        graph_context_relevancy_metric_reason = graph_context_relevancy_metric.reason

        # Custom logic to set score
        composite_score = min(relevancy_score, faithfulness_score, graph_context_recall_metric_score, graph_context_relevancy_metric_score)
        self.score = 0 if self.strict_mode and composite_score < self.threshold else composite_score

        # Custom logic to set reason
        if self.include_reason:
            self.reason = relevancy_reason + "\n" + faithfulness_reason + "\n" + graph_context_recall_metric_reason + "\n" + graph_context_relevancy_metric_reason

        # Custom logic to set success
        self.success = self.score >= self.threshold
        
class GraphContextualRecall(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        
    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
    ) -> float:
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            self.verdicts: List[ContextualRecallVerdict] = (
                self._generate_verdicts(
                    test_case.expected_output, test_case.retrieval_context, test_case.additional_metadata
                )
            )
            self.score = self._calculate_score()
            self.reason = self._generate_reason(test_case.expected_output)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    def _generate_reason(self, expected_output: str):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = GraphContextualRecallTemplate.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
            score=format(self.score, ".2f"),
        )

        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        justified_sentences = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                justified_sentences += 1

        score = justified_sentences / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_verdicts(
        self, expected_output: str, retrieval_context: List[str], cypher_query: Optional[str] = None
    ) -> List[ContextualRecallVerdict]:
        prompt = GraphContextualRecallTemplate.generate_verdicts(
            expected_output=expected_output, retrieval_context=retrieval_context, cypher_query=cypher_query
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [
                ContextualRecallVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts: Verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    ContextualRecallVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Graph Contextual Recall"

class GraphContextualRecallTemplate:
    @staticmethod
    def generate_reason(
        expected_output, supportive_reasons, unsupportive_reasons, score
    ):
        return f"""
Given the original expected output, a list of supportive reasons, and a list of unsupportive reasons (which is deduced directly from the 'expected output'), and a contextual recall score (closer to 1 the better), summarize a CONCISE reason for the score.
A supportive reason is the reason why a certain sentence in the original expected output can be attributed to the node in the retrieval context.
An unsupportive reason is the reason why a certain sentence in the original expected output cannot be attributed to anything in the retrieval context.
In your reason, you should related supportive/unsupportive reasons to the sentence number in expected output, and info regarding the node number in retrieval context to support your final reason. The first mention of "node(s)" should specify "node(s) in retrieval context)".

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <contextual_recall_score> because <your_reason>."
}}

DO NOT mention 'supportive reasons' and 'unsupportive reasons' in your reason, these terms are just here for you to understand the broader scope of things.
If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
**

Contextual Recall Score:
{score}

Expected Output:
{expected_output}

Supportive Reasons:
{supportive_reasons}

Unsupportive Reasons:
{unsupportive_reasons}

JSON:
"""

    @staticmethod
    def generate_verdicts(expected_output, retrieval_context, cypher_query):
        return f"""
For EACH sentence in the given expected output below, determine whether the sentence can be attributed to the nodes of retrieval contexts that generated from the cypher query. Please generate a list of JSON with two keys: `verdict` and `reason`.
The `verdict` key should STRICTLY be either a 'yes' or 'no'. Answer 'yes' if the sentence can be attributed to any parts of the retrieval context and the cypher query, else answer 'no'.
The `reason` key should provide a reason why to the verdict. In the reason, you should aim to include the node(s) count in the retrieval context (eg., 1st node, and 2nd node in the retrieval context) that is attributed to said sentence. You should also aim to quote the specific part of the retrieval context to justify your verdict, but keep it extremely concise and cut short the quote with an ellipsis if possible. 


**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects, each with two keys: `verdict` and `reason`.

{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
        ...
    ]  
}}

Since you are going to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE STRICTLY EQUAL to the number of sentences in of `expected output`.
**

Expected Output:
{expected_output}

Cypher Query:
{cypher_query}

Retrieval Context:
{retrieval_context}

JSON:
"""

class GraphContextualRelevancy(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        
    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[0]
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            self.verdicts_list: List[ContextualRelevancyVerdicts] = [
                (self._generate_verdicts(test_case.input, context, test_case.additional_metadata))
                for context in test_case.retrieval_context
            ]
            self.score = self._calculate_score()
            self.reason = self._generate_reason(test_case.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = construct_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{prettify_list(self.verdicts_list)}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )

            return self.score

    def _generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        relevant_statements = []
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                if verdict.verdict.lower() == "no":
                    irrelevancies.append(verdict.reason)
                else:
                    relevant_statements.append(verdict.statement)

        prompt: dict = GraphContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            relevant_statements=relevant_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _calculate_score(self):
        total_verdicts = 0
        relevant_statements = 0
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                total_verdicts += 1
                if verdict.verdict.lower() == "yes":
                    relevant_statements += 1

        if total_verdicts == 0:
            return 0

        score = relevant_statements / total_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _generate_verdicts(
        self, input: str, context: str, cypher_query: Optional[str] = None
    ) -> ContextualRelevancyVerdicts:
        prompt = GraphContextualRelevancyTemplate.generate_verdicts(
            input=input, context=context, cypher_query=cypher_query
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return ContextualRelevancyVerdicts(**data)
        else:
            try:
                res = self.model.generate(
                    prompt, schema=ContextualRelevancyVerdicts
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return ContextualRelevancyVerdicts(**data)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Graph Contextual Relevancy"

class GraphContextualRelevancyTemplate:
    @staticmethod
    def generate_reason(
        input: str,
        irrelevancies: List[str],
        relevant_statements: List[str],
        score: float,
    ):
        return f"""Based on the given input, reasons for why the retrieval context is irrelevant to the input, the statements in the retrieval context that is actually relevant to the retrieval context, and the contextual relevancy score (the closer to 1 the better), please generate a CONCISE reason for the score.
In your reason, you should quote data provided in the reasons for irrelevancy and relevant statements to support your point.

** 
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <contextual_relevancy_score> because <your_reason>."
}}

If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).
**


Contextual Relevancy Score:
{score}

Input:
{input}

Reasons for why the retrieval context is irrelevant to the input:
{irrelevancies}

Statement in the retrieval context that is relevant to the input:
{relevant_statements}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, context: str, cypher_query):
        return f"""Based on the input and context (cypher and query output), please generate a JSON object to indicate whether each statement found in the context is relevant to the provided input. The JSON will be a list of 'verdicts', with 2 mandatory fields: 'verdict' and 'statement', and 1 optional field: 'reason'.
You should first extract statements found in the context, which are high level information found in the context, before deciding on a verdict and optionally a reason for each statement.
The 'verdict' key should STRICTLY be either 'yes' or 'no', and states whether the statement is relevant to the input.
Provide a 'reason' ONLY IF verdict is no. You MUST quote the irrelevant parts of the statement to back up your reason.

**
IMPORTANT: Please make sure to only return in JSON format.
Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "What were some of Einstein's achievements?"

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "statement": "Einstein won the Nobel Prize for his discovery of the photoelectric effect in 1968",
        }},
        {{
            "verdict": "no",
            "statement": "There was a cat.",
            "reason": "The retrieval context contained the information 'There was a cat' when it has nothing to do with Einstein's achievements."
        }}
    ]
}}
**

Input:
{input}

Cypher Query:
{cypher_query}

Context:
{context}

JSON:
"""
