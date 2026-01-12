import logging
import pandas as pd
from typing import List, Optional, Any, Dict

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.schema import Trace
from observix.evaluation.trace_utils import extract_eval_params

import os

logger = logging.getLogger(__name__)

def _set_env_from_kwargs(kwargs):
    if kwargs.get("api_key"):
        # Phoenix OpenAIModel uses OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = kwargs["api_key"]
        # If user provides Azure keys, mapped similarly
        if kwargs.get("azure_endpoint"):
             os.environ["AZURE_API_BASE"] = kwargs["azure_endpoint"]
             os.environ["AZURE_OPENAI_API_KEY"] = kwargs["api_key"] # Phoenix uses this or OPENAI_API_KEY depending on config
             # We might need to map specific Phoenix Azure env vars if different
             # But generic OpenAIModel works with OPENAI_API_KEY
    
    # Also handle AZURE specific if passed explicitly
    if kwargs.get("api_version"):
        os.environ["AZURE_API_VERSION"] = kwargs["api_version"]
    if kwargs.get("deployment_name"):
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = kwargs["deployment_name"]

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
        TOOL_CALLING_PROMPT_TEMPLATE
    )
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    OpenAIModel = object # Mock for type hint if needed
    LangChainModel = object

def _wrap_llm_for_phoenix(llm: Any, default_model_name: str = "gpt-4") -> Any:
    """
    Wraps an LLM object (LangChain Runnable or OpenAI Client) into a Phoenix Model.
    """
    if not PHOENIX_AVAILABLE:
        return None

    # Check for LangChain Runnable (has invoke/batch)
    if hasattr(llm, "invoke") or hasattr(llm, "batch"):
        return LangChainModel(model=llm)

    # Check for OpenAI Client (has chat.completions)
    # Phoenix OpenAIModel typically creates its own client, but we can try to use it 
    # if we can configure it. However, standard OpenAIModel takes model name.
    # If we have a raw client, it's hard to inject into OpenAIModel directly without
    # using a custom adapter or relying on env vars the client also used.
    # But often users just want to control the model name.
    # If the user passed a client, we might need to assume environment is set or 
    # use a generic wrapper.
    # Ideally we'd use the client properties. 
    # For now, if it looks like an OpenAI client, we might try to just Create OpenAIModel 
    # with specific settings if possible, otherwise we fallback to default OpenAIModel usage
    # but that ignores the passed client instance.
    # A better approach for raw client might be to just not wrap it if Phoenix doesn't support it 
    # and rely on global config, BUT user explicitly passed it.
    
    # As a fallback for raw clients, we can try to assume it's OpenAI-compatible 
    # and if Phoenix supports passing client (some versions do), use it. 
    # If not, we fall back to creating OpenAIModel which uses env vars.
    return OpenAIModel(model=default_model_name)


class PhoenixMetricEvaluator(Evaluator):
    """
    Evaluator using Arize Phoenix (arize-phoenix-evals) metrics.
    """
    def __init__(self, model: Any, template: str, rails_map: Dict[str, Any], name: str, input_column_map: Dict[str, str], llm: Optional[Any] = None, **kwargs):
        """
        Args:
            model: The Phoenix Model wrapper (e.g. OpenAIModel)
            template: The prompt template string
            rails_map: The rails map for parsing output
            name: The metric name
            input_column_map: Mapping from standard args (input_query, output, expected, context) to dataframe columns expected by template.
                               Key: standard arg name, Value: dataframe column name
            llm: Optional underlying LLM object to wrap if model is not provided.
        """
        if not PHOENIX_AVAILABLE:
            raise ImportError(
                "arize-phoenix-evals is not installed. Please install it with "
                "`pip install arize-phoenix-evals` or `uv sync --extra eval`."
            )
            
        if llm and model is None:
            model = _wrap_llm_for_phoenix(llm)

        self.model = model
        self.template = template
        self.rails_map = rails_map
        self._name = name
        self.input_column_map = input_column_map

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
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
                
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                reason=f"Label: {label}. Explanation: {explanation}" if explanation else f"Label: {label}"
            )

        except Exception as e:
            logger.error(f"Phoenix evaluation failed: {e}")
            raise e

class PhoenixHallucinationEvaluator(PhoenixMetricEvaluator):
    def __init__(self, model=None, llm: Optional[Any] = None, **kwargs):
        _set_env_from_kwargs(kwargs)
        if model is None and llm is None:
             from phoenix.evals import OpenAIModel
             model = OpenAIModel(model="gpt-4")
        
        # If llm is provided, PhoenixMetricEvaluator will handle wrapping if model is None
             
        super().__init__(
            model=model,
            template=HALLUCINATION_PROMPT_TEMPLATE,
            rails_map=HALLUCINATION_PROMPT_RAILS_MAP,
            name="phoenix_hallucination",
            input_column_map={
                "output": "output",
                "input_query": "input", 
                "context": "context"
            },
            llm=llm,
            **kwargs
        )
        self.input_column_map = {
             "input_query": "input",
             "output": "output",
             "context": "reference" 
        }

class PhoenixQAEvaluator(PhoenixMetricEvaluator):
    def __init__(self, model=None, llm: Optional[Any] = None, **kwargs):
        _set_env_from_kwargs(kwargs)
        if model is None and llm is None:
             from phoenix.evals import OpenAIModel
             model = OpenAIModel(model="gpt-4")
        super().__init__(
            model=model,
            template=QA_PROMPT_TEMPLATE,
            rails_map=QA_PROMPT_RAILS_MAP,
            name="phoenix_qa",
             input_column_map={
                 "input_query": "input",
                 "output": "output",
                 "expected": "reference"
            },
            llm=llm,
            **kwargs
        )

class PhoenixRAGRelevancyEvaluator(PhoenixMetricEvaluator):
    def __init__(self, model=None, llm: Optional[Any] = None, **kwargs):
        _set_env_from_kwargs(kwargs)
        if model is None and llm is None:
             from phoenix.evals import OpenAIModel
             model = OpenAIModel(model="gpt-4")
        super().__init__(
            model=model,
            template=RAG_RELEVANCY_PROMPT_TEMPLATE,
            rails_map=RAG_RELEVANCY_PROMPT_RAILS_MAP,
            name="phoenix_rag_relevancy",
             input_column_map={
                 "input_query": "input",
                 "expected": "reference"
            },
            llm=llm,
            **kwargs
        )
        self.input_column_map = {
            "input_query": "input",
            "context": "reference" 
        }

class PhoenixAgentFunctionCalling(PhoenixMetricEvaluator):
    def __init__(self, model=None, llm: Optional[Any] = None, **kwargs):
        _set_env_from_kwargs(kwargs)
        if model is None and llm is None:
             from phoenix.evals import OpenAIModel
             model = OpenAIModel(model="gpt-4")
        super().__init__(
            model=model,
            template=TOOL_CALLING_PROMPT_TEMPLATE,
            rails_map=TOOL_CALLING_PROMPT_RAILS_MAP,
            name="phoenix_agent_function_calling",
             input_column_map={
                 "input_query": "input",
                 "expected": "reference" 
            },
            llm=llm,
            **kwargs
        )
        self.input_column_map = {
            "input_query": "question",
            "output": "tool_call",
            "context": "tool_definitions" 
        }

