import logging
import pandas as pd
from typing import List, Optional, Any, Dict

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.schema import Trace
from observix.evaluation.trace_utils import extract_eval_params
from observix import observe

import os

from observix.llm import get_llm

logger = logging.getLogger(__name__)


try:
    from phoenix.evals import (
        llm_classify, 
        OpenAIModel,
        LangChainModel,
        HALLUCINATION_PROMPT_TEMPLATE,
        HALLUCINATION_PROMPT_RAILS_MAP,
        QA_PROMPT_TEMPLATE,
        QA_PROMPT_RAILS_MAP,
        RAG_RELEVANCY_PROMPT_TEMPLATE,
        RAG_RELEVANCY_PROMPT_RAILS_MAP,
        TOOL_CALLING_PROMPT_RAILS_MAP,
        TOOL_CALLING_PROMPT_TEMPLATE,
        TOXICITY_PROMPT_TEMPLATE,
        TOXICITY_PROMPT_RAILS_MAP
    )
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    OpenAIModel = object # Mock for type hint if needed
    LangChainModel = object

class PhoenixCustomModel:
    """
    Wrapper for Phoenix Models to handle dynamic configuration like DeepEval/Ragas custom models.
    """
    def __init__(
        self,
        provider: str,
        model_name: str = "gpt-4",
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs
        self._phoenix_model = self._get_model()

    def _get_model(self):
        if not PHOENIX_AVAILABLE:
            return None

        # Map 'langchain' provider (which evaluation used for Groq) to 'groq' for factory
        internal_provider = self.provider
        if self.provider == "langchain":
            internal_provider = "groq"
        
        full_model = f"{internal_provider}/{self.model_name}" if self.model_name else internal_provider

        llm = get_llm(
            model=full_model,
            framework="langchain" if self.provider == "langchain" else "openai",
            **self.kwargs
        )
        
        if self.provider == "langchain":
            return LangChainModel(model=llm)
        else:
            # Phoenix OpenAIModel usually takes the model name and uses its own client logic
            # but we want our traced client. However, Phoenix OpenAIModel might be strict.
            # If so, we might need to keep using our wrapped client if possible.
            return OpenAIModel(model=self.model_name)

    @property
    def model(self):
        return self._phoenix_model


def _wrap_llm_for_phoenix(llm: Any, default_model_name: str = "gpt-4") -> Any:
    """
    Wraps an LLM object (LangChain Runnable or OpenAI Client) into a Phoenix Model.
    """
    if not PHOENIX_AVAILABLE:
        return None

    # Check for LangChain Runnable (has invoke/batch)
    if hasattr(llm, "invoke") or hasattr(llm, "batch"):
        return LangChainModel(model=llm)

    # Fallback to OpenAIModel
    return OpenAIModel(model=default_model_name)


class PhoenixMetricEvaluator(Evaluator):
    """
    Evaluator using Arize Phoenix (arize-phoenix-evals) metrics.
    """
    def __init__(
        self, 
        template: str, 
        rails_map: Dict[str, Any], 
        name: str, 
        input_column_map: Dict[str, str], 
        model: Optional[Any] = None, # direct phoenix model (legacy)
        llm: Optional[Any] = None, # direct llm object (legacy)
        provider: Optional[str] = None, # new standard
        model_name: Optional[str] = None, # new standard
        **kwargs
    ):
        if not PHOENIX_AVAILABLE:
            raise ImportError(
                "arize-phoenix-evals is not installed. Please install it with "
                "`pip install arize-phoenix-evals` or `uv sync --extra eval`."
            )
            
        # Priority 1: Use provided Phoenix Model or LLM object
        if model:
            self.model = model
        elif llm:
            self.model = _wrap_llm_for_phoenix(llm)
        else:
            # Priority 2: Use Provider/Model Config
            provider = provider or "openai"
            model_name = model_name or "gpt-4"
            self.custom_model = PhoenixCustomModel(provider=provider, model_name=model_name, **kwargs)
            self.model = self.custom_model.model

        self.template = template
        self.rails_map = rails_map
        self._name = name
        self.input_column_map = input_column_map

    @property
    def name(self) -> str:
        return self._name

    # @observe removed as tracing is now handled in base class
    def _evaluate(
        self,
        output: str = "",
        expected: Optional[str] = None,
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        
        # Handle Trace object
        trace = kwargs.get("trace")
        if isinstance(trace, Trace):
            params = extract_eval_params(trace)
            # Use extracted params if not explicitly provided
            if not output: output = params.get("output", "")
            if not input_query: input_query = params.get("input_query", "")
            if not context: context = params.get("context", [])
        
        # Prepare Dataframe Row
        data = {}
        
        # Map inputs
        if "output" in self.input_column_map:
            data[self.input_column_map["output"]] = output
        if "input_query" in self.input_column_map:
            data[self.input_column_map["input_query"]] = input_query or ""
        if "expected" in self.input_column_map:
            data[self.input_column_map["expected"]] = expected or ""
        if "context" in self.input_column_map:
             # Phoenix usually expects text for retrieval, so join list
            data[self.input_column_map["context"]] = "\n\n".join(context) if context else ""
            
        df = pd.DataFrame([data])
        
        try:
            rails = list(self.rails_map.values())
            
            result_df = llm_classify(
                dataframe=df,
                model=self.model,
                template=self.template,
                rails=rails,
                provide_explanation=True 
            )
            
            label = result_df['label'][0]
            explanation = result_df['explanation'][0] if 'explanation' in result_df.columns else None
            
            score = 1.0 
            passed = True
            
            if str(label).lower() in ["correct", "factual", "relevant"]:
                score = 1.0
                passed = True
            elif str(label).lower() in ["incorrect", "hallucinated", "irrelevant"]:
                score = 0.0
                passed = False
            else:
                score = 0.0 
                passed = False 
                
            # Extract Trace ID from current context if available
            from opentelemetry import trace as otel_trace
            current_span = otel_trace.get_current_span()
            trace_id_hex = None
            
            if current_span.get_span_context().is_valid:
                trace_id_hex = f"{current_span.get_span_context().trace_id:032x}"

            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                reason=f"Label: {label}. Explanation: {explanation}" if explanation else f"Label: {label}",
                metadata={"trace_id": trace_id_hex} if trace_id_hex else {}
            )

        except Exception as e:
            logger.error(f"Phoenix evaluation failed: {e}")
            raise e

class PhoenixHallucinationEvaluator(PhoenixMetricEvaluator):
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(
            template=HALLUCINATION_PROMPT_TEMPLATE,
            rails_map=HALLUCINATION_PROMPT_RAILS_MAP,
            name="phoenix_hallucination",
            input_column_map={
                "output": "output",
                "input_query": "input", 
                "context": "context"
            },
            provider=provider,
            model_name=model,
            **kwargs
        )
        self.input_column_map = {
             "input_query": "input",
             "output": "output",
             "context": "reference" 
        }

class PhoenixQAEvaluator(PhoenixMetricEvaluator):
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(
            template=QA_PROMPT_TEMPLATE,
            rails_map=QA_PROMPT_RAILS_MAP,
            name="phoenix_qa",
             input_column_map={
                 "input_query": "input",
                 "output": "output",
                 "expected": "reference"
            },
            provider=provider,
            model_name=model,
            **kwargs
        )

class PhoenixRAGRelevancyEvaluator(PhoenixMetricEvaluator):
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(
            template=RAG_RELEVANCY_PROMPT_TEMPLATE,
            rails_map=RAG_RELEVANCY_PROMPT_RAILS_MAP,
            name="phoenix_rag_relevancy",
             input_column_map={
                 "input_query": "input",
                 "expected": "reference"
            },
            provider=provider,
            model_name=model,
            **kwargs
        )
        self.input_column_map = {
            "input_query": "input",
            "context": "reference" 
        }

class PhoenixAgentFunctionCalling(PhoenixMetricEvaluator):
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(
            template=TOOL_CALLING_PROMPT_TEMPLATE,
            rails_map=TOOL_CALLING_PROMPT_RAILS_MAP,
            name="phoenix_agent_function_calling",
             input_column_map={
                 "input_query": "input",
                 "expected": "reference" 
            },
            provider=provider,
            model_name=model,
            **kwargs
        )
        self.input_column_map = {
            "input_query": "question",
            "output": "tool_call",
            "context": "tool_definitions" 
        }

class PhoenixToxicityEvaluator(PhoenixMetricEvaluator):
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(
            template=TOXICITY_PROMPT_TEMPLATE,
            rails_map=TOXICITY_PROMPT_RAILS_MAP,
            name="phoenix_toxicity",
             input_column_map={
                 "input_query": "input",
                 "output": "output",
                 "context": "reference" 
            },
            provider=provider,
            model_name=model,
            **kwargs
        )
