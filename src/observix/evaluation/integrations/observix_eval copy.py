import os
import re
import logging
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_core.messages import HumanMessage

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.schema import Trace
from observix.evaluation.trace_utils import trace_to_text, extract_eval_params, extract_tool_calls, extract_workflow_details
from observix.evaluation.integrations.prompts import (
    TOOL_SELECTION_PROMPT_TEMPLATE,
    TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE,
    TOOL_SEQUENCE_PROMPT_TEMPLATE,
    AGENT_ROUTING_PROMPT_TEMPLATE,
    HITL_PROMPT_TEMPLATE,
    WORKFLOW_COMPLETION_PROMPT_TEMPLATE,
    CUSTOM_METRIC_PROMPT_TEMPLATE,
)

load_dotenv()
logger = logging.getLogger(__name__)

class ObservixEvaluator(Evaluator):
    def __init__(self, metric_name: str, provider: str, model: str, prompt_template: str, **kwargs):
        self.metric_name = metric_name
        self.provider = provider
        self.model_name = model
        self.prompt_template = prompt_template
        self.temperature = kwargs.get("temperature", 0.1)

        # Setup environment variables for LLM providers
        api_key = kwargs.get("api_key")
        if api_key:
            if provider == "azure":
                os.environ["AZURE_OPENAI_KEY"] = api_key
            elif provider == "langchain":
                os.environ["GROQ_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = api_key

        if kwargs.get("azure_endpoint"):
            os.environ["AZURE_API_BASE"] = kwargs["azure_endpoint"]

        if kwargs.get("api_version"):
            os.environ["AZURE_API_VERSION"] = kwargs["api_version"]

        if kwargs.get("deployment_name"):
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = kwargs["deployment_name"]

        self.llm = self._get_llm()

    def _get_llm(self):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(400, "OPENAI_API_KEY is required")

            from observix.llm.openai import OpenAI
            return OpenAI(name=self.metric_name, api_key=api_key)

        elif self.provider == "azure":
            api_base = os.getenv("AZURE_API_BASE")
            api_version = os.getenv("AZURE_API_VERSION")
            api_key = os.getenv("AZURE_OPENAI_KEY")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

            if not all([api_base, api_version, api_key, deployment]):
                raise HTTPException(
                    400,
                    "Azure requires AZURE_API_BASE, AZURE_API_VERSION, "
                    "AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME",
                )

            from observix.llm.openai import AzureOpenAI
            return AzureOpenAI(
                name=self.metric_name,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )

        elif self.provider == "langchain":
            api_key = os.getenv("GROQ_API_KEY")
            model = self.model_name or "openai/gpt-oss-120b"

            if not api_key:
                raise HTTPException(400, "GROQ_API_KEY is required")

            from observix.llm.langchain import ChatGroq
            return ChatGroq(
                model=model,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=2500,
            )

        raise HTTPException(400, f"Unsupported provider: {self.provider}")

    @property
    def name(self) -> str:
        return self.metric_name

    async def _generate_response(self, prompt: str) -> str:
        if self.provider in {"openai", "azure"}:
            response = self.llm.chat.completions.create(
                model=self.model_name or (self.llm.deployment_name if hasattr(self.llm, "deployment_name") else "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        elif self.provider == "langchain":
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content

        raise RuntimeError("LLM provider interface mismatch")

    def _parse_score(self, response: str) -> float:
        # Attempt to find a float between 0 and 1 in the response
        try:
            # Look for explicit pattern "Score: X" or just numbers
            # The prompt asks for a rate on a scale from 0 to 1.
            # We look for the last number in the text which is likely the score.
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            if not matches:
                return 0.0
            
            # Take the last number found as it's often at the end "Score: 0.8"
            score = float(matches[-1])
            
            # Normalize if needed (though prompt asks for 0-1)
            if score > 1.0 and score <= 10.0:
                 score = score / 10.0
            elif score > 10.0:
                 score = score / 100.0 # Just a safe fallback
                 
            return min(max(score, 0.0), 1.0)
        except Exception:
            logger.warning(f"Could not parse score from response: {response}")
            return 0.0

    async def evaluate(
        self,
        **kwargs,
    ) -> EvaluationResult:
        
        trace = kwargs.get("trace")
        trace_str = ""
        param_dict = {}


        params = extract_eval_params(trace)
        # Use the new extraction logic
        workflow_details = extract_workflow_details(trace)
        
        output = output or params.get("output", "")
        input_query = input_query or params.get("input_query", "")
        context = context or params.get("context", [])
        trace_str = trace_to_text(trace)
        
        # Extract additional template variables
        param_dict["trace"] = trace_str
        param_dict["question"] = input_query
        
        # Auto-populate agent/tool info from trace if not provided
        if "agent_definitions" not in kwargs:
                agents = workflow_details["agents"]
                if agents:
                    desc = "\n".join([f"- {a['name']}: {a['input'][:100]}..." for a in agents])
                    param_dict["agent_definitions"] = desc
                else:
                    param_dict["agent_definitions"] = "No agents detected in trace."

        if "tool_definitions" not in kwargs:
                tools = workflow_details["tools"]
                if tools:
                    desc = "\n".join([f"- {t['name']}: {t['input'][:100]}..." for t in tools])
                    param_dict["tool_definitions"] = desc
                else:
                    param_dict["tool_definitions"] = "No tools detected in trace."

        # Tool/Workflow Sequence
        if "{tool_sequence}" in self.prompt_template:
                param_dict["tool_sequence"] = " -> ".join(workflow_details["sequence"]) if workflow_details["sequence"] else "No sequence found."
        
        if "{tool_call}" in self.prompt_template:
            tools = workflow_details["tools"]
            param_dict["tool_call"] = "\n".join([f"{t['name']}: {t['input']}" for t in tools]) if tools else "No tool calls found."
        
        # Pass through any other kwargs as template variables
        param_dict.update(kwargs)
        # Ensure defaults for missing keys to avoid format errors
        for key in ["tool_definitions", "agent_definitions", "HITL_INFO", "custom_instructions"]:
            if key not in param_dict:
                param_dict[key] = kwargs.get(key, "N/A")

        try:
            # Format prompt safely
            try:
                formatted_prompt = self.prompt_template.format(**param_dict)
            except KeyError as e:
                # Fallback if keys are missing from trace/kwargs
                logger.warning(f"Missing key for prompt formatting: {e}")
                # Try to fill missing keys with placeholders
                missing_key = str(e).strip("'")
                param_dict[missing_key] = "N/A"
                formatted_prompt = self.prompt_template.format(**param_dict)

            response_text = await self._generate_response(formatted_prompt)
            score = self._parse_score(response_text)
            passed = score >= kwargs.get("threshold", 0.5)

            # Metadata
            trace_id_hex = None
            if trace:
                 trace_id_hex = trace.trace_id

            return EvaluationResult(
                metric_name=self.metric_name,
                score=score,
                passed=passed,
                reason=response_text,
                metadata={"trace_id": trace_id_hex} if trace_id_hex else {}
            )

        except Exception as e:
            logger.exception(f"Observix custom evaluation failed: {e}")
            raise


class ToolSelectionEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("ToolSelection", provider, model, TOOL_SELECTION_PROMPT_TEMPLATE, **kwargs)

class ToolInputStructureEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("ToolInputStructure", provider, model, TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE, **kwargs)

class ToolSequenceEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("ToolSequence", provider, model, TOOL_SEQUENCE_PROMPT_TEMPLATE, **kwargs)

class AgentRoutingEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("AgentRouting", provider, model, AGENT_ROUTING_PROMPT_TEMPLATE, **kwargs)

class HITLEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("HITL", provider, model, HITL_PROMPT_TEMPLATE, **kwargs)

class WorkflowCompletionEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("WorkflowCompletion", provider, model, WORKFLOW_COMPLETION_PROMPT_TEMPLATE, **kwargs)

class CustomEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("CustomMetric", provider, model, CUSTOM_METRIC_PROMPT_TEMPLATE, **kwargs)
class AccuracyEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
         # Accuracy is often task-specific. Reusing WorkflowTemplate or Custom if no specific template exists. 
         # Assuming Custom structure for now or generic query check.
         # The User previously mentioned "Accuracy" but didn't provide a template. 
         # We will use CUSTOM_METRIC_PROMPT_TEMPLATE but name it Accuracy.
         super().__init__("Accuracy", provider, model, CUSTOM_METRIC_PROMPT_TEMPLATE, **kwargs)
