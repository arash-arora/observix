import os
import logging
from typing import Optional, List, Any
from dotenv import load_dotenv

# Load env vars
load_dotenv()

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.evaluation.trace_utils import trace_to_text, extract_eval_params
from observix.schema import Trace
from observix.evaluation.integrations.prompts import (
    TOOL_SELECTION_PROMPT_TEMPLATE,
    TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE,
    TOOL_SEQUENCE_PROMPT_TEMPLATE,
    AGENT_ROUTING_PROMPT_TEMPLATE,
    HITL_PROMPT_TEMPLATE,
    WORKFLOW_COMPLETION_PROMPT_TEMPLATE,
    CUSTOM_METRIC_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

# Try importing OpenAI client
try:
    from observix.llm.openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = object
    AzureOpenAI = object

class ObservixEvalLLM:
    """
    Wrapper for LLM clients (OpenAI, Azure, Groq) to unify generation.
    Can also wrap LangChain Runnable or raw OpenAI-compatible client.
    """
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_type: str = "openai", # openai, azure, groq
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        client_instance: Optional[Any] = None
    ):
        self.model = model
        self.client_type = client_type.lower()
        self.client_instance = client_instance
        self.is_langchain = False

        if self.client_instance:
            # If client is provided, detect type
            if hasattr(self.client_instance, "invoke"):
                self.is_langchain = True
            elif hasattr(self.client_instance, "chat"):
                # OpenAI compatible client
                self.client = self.client_instance
            else:
                 # Assume generic client or maybe user passed something else
                 self.client = self.client_instance
        else:
             # Create client as before
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is required. Install with `pip install openai`.")

            if self.client_type == "groq":
                 # Groq is OpenAI compatible
                 self.client = OpenAI(
                     base_url=base_url or "https://api.groq.com/openai/v1",
                     api_key=api_key or os.getenv("GROQ_API_KEY")
                 )
                 if not self.model: 
                     self.model = "llama-3.3-70b-versatile"
                     
            elif self.client_type == "azure":
                self.client = AzureOpenAI(
                    api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version or "2024-02-01",
                    azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                    deployment_name=azure_deployment or model
                )
                
            else: # openai
                self.client = OpenAI(
                    api_key=api_key or os.getenv("OPENAI_API_KEY"),
                    base_url=base_url
                )

    def generate(self, prompt: str) -> str:
        try:
            if self.is_langchain:
                # LangChain usage
                response = self.client_instance.invoke(prompt)
                # Response might be a string or AIMessage
                if hasattr(response, "content"):
                    return response.content
                return str(response)
            else:
                # OpenAI usage
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant for evaluation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            raise e

class ObservixEvalEvaluator(Evaluator):
    """
    Base class for ObsEval metrics using LLM.
    """
    def __init__(self, metric_name: str, prompt_template: str, llm: Any):
        self._name = metric_name
        self.prompt_template = prompt_template
        
        # Ensure llm is ObservixEvalLLM
        if isinstance(llm, ObservixEvalLLM):
            self.llm = llm
        else:
            # Wrap raw client
            self.llm = ObservixEvalLLM(client_instance=llm, model="gpt-4o") # default model if not known

    @property
    def name(self) -> str:
        return self._name

    def _parse_score(self, response: str) -> float:
        # Heuristic to find the score 0 to 1
        try:
            import re
            # Find all numbers
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            if matches:
                # Check in reverse for a number between 0 and 1
                for match in reversed(matches):
                    try:
                        val = float(match)
                        if 0 <= val <= 1:
                            return val
                    except:
                        continue
            return 0.0 # Fallback
        except:
            return 0.0

    def evaluate(
        self,
        output: str = "",        # Mapped from kwargs usually
        expected: Optional[str] = None,
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        
        # We need to construct the prompt arguments.
        # The base Evaluator signature is fixed, but custom metrics need:
        # trace, tool_call, tool_definitions, agent_definitions, etc.
        # We expect these in `kwargs` or mapped from standard args.
        
        # Standard mappings:
        # input_query -> {question}
        # output -> {tool_call}, {tool_sequence}, etc?
        # context -> {trace}? or passed explicitly in kwargs?
        
        # We'll allow kwargs to override or supply missing variables.
        
        # Prepare variables dict
        try:
            # Handle Trace object in kwargs
            trace_input = kwargs.get("trace", "")
            trace_text = ""
            
            if isinstance(trace_input, Trace):
                # 1. Convert to text for {trace} prompt variable
                trace_text = trace_to_text(trace_input)
                
                # 2. Extract standard params if missing
                params = extract_eval_params(trace_input)
                if not input_query: input_query = params.get("input_query", "")
                if not output: output = params.get("output", "")
                
            else:
                trace_text = str(trace_input)

            # 1. Start with known args
            format_args = {
                "question": input_query or "",
                "tool_call": output or "",
                "tool_sequence": output or "", # Reuse output if sequence
                "trace": trace_text,
                "tool_definitions": kwargs.get("tool_definitions", ""),
                "agent_definitions": kwargs.get("agent_definitions", ""),
                "HITL_INFO": kwargs.get("HITL_INFO", ""),
                "custom_instructions": kwargs.get("custom_instructions", "")
            }
            
            # 2. Format
            prompt = self.prompt_template.format(**format_args)
            
            # 3. Call LLM
            response = self.llm.generate(prompt)
            
            # 4. Parse
            score = self._parse_score(response)
            passed = score >= 0.5 # Threshold?
            
            # Extract Trace ID
            from opentelemetry import trace as otel_trace
            current_span = otel_trace.get_current_span()
            trace_id_hex = None
            if current_span.get_span_context().is_valid:
                trace_id_hex = f"{current_span.get_span_context().trace_id:032x}"

            return EvaluationResult(
                metric_name=self.name,
                score=score,
                passed=passed,
                reason=response,
                metadata={"trace_id": trace_id_hex} if trace_id_hex else {}
            )
            
        except KeyError as e:
            return EvaluationResult(
                 metric_name=self.name,
                 score=0.0,
                 passed=False,
                 reason=f"Missing variable for prompt: {e}"
            )
        except Exception as e:
             return EvaluationResult(
                 metric_name=self.name,
                 score=0.0,
                 passed=False,
                 reason=f"Error: {e}"
            )

# Specific Evaluators
class ToolSelectionEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("tool_selection", TOOL_SELECTION_PROMPT_TEMPLATE, llm)

class ToolInputStructureEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("tool_input_structure", TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE, llm)

class ToolSequenceEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("tool_sequence", TOOL_SEQUENCE_PROMPT_TEMPLATE, llm)

class AgentRoutingEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("agent_routing", AGENT_ROUTING_PROMPT_TEMPLATE, llm)

class HITLEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("hitl_evaluation", HITL_PROMPT_TEMPLATE, llm)

class WorkflowCompletionEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("workflow_completion", WORKFLOW_COMPLETION_PROMPT_TEMPLATE, llm)

class CustomMetricEvaluator(ObservixEvalEvaluator):
    def __init__(self, llm: Any):
        super().__init__("custom_metric", CUSTOM_METRIC_PROMPT_TEMPLATE, llm)

# Unified Facade (Optional)
class ObservixEval:
    def __init__(
        self,
        client_type: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None
    ):
        self.llm = ObservixEvalLLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client_type=client_type,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version
        )
        
    def get_evaluator(self, metric_type: str) -> Evaluator:
        metrics = {
            "tool_selection": ToolSelectionEvaluator,
            "tool_input_structure": ToolInputStructureEvaluator,
            "tool_sequence": ToolSequenceEvaluator,
            "agent_routing": AgentRoutingEvaluator,
            "hitl": HITLEvaluator,
            "workflow_completion": WorkflowCompletionEvaluator,
            "custom": CustomMetricEvaluator
        }
        if metric_type not in metrics:
            raise ValueError(f"Unknown metric type: {metric_type}")
        return metrics[metric_type](self.llm)

