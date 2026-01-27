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
from observix.llm import get_llm
from observix.evaluation.integrations.trace_sanitizer import TraceSanitizer

load_dotenv()
logger = logging.getLogger(__name__)

class ObservixEvaluator(Evaluator):
    def __init__(self, metric_name: str, provider: str, model: str, prompt_template: str, **kwargs):
        self.metric_name = metric_name
        self.provider = provider
        self.model_name = model
        self.prompt_template = prompt_template
        self.temperature = kwargs.get("temperature", 0.1)

        # Map 'langchain' provider (which evaluation used for Groq) to 'groq' for factory
        internal_provider = provider
        if provider == "langchain":
            internal_provider = "groq"
        
        full_model = f"{internal_provider}/{model}" if model else internal_provider

        self.llm = get_llm(
            model=full_model,
            temperature=self.temperature,
            framework="langchain" if provider == "langchain" else "openai",
            name=self.metric_name,
            **kwargs
        )
        self.sanitizer = TraceSanitizer(provider=provider, model=model, **kwargs)

    def _get_llm(self):
        # Deprecated, maintained for internal use if any
        return self.llm


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

    async def _evaluate(
        self,
        **kwargs,
    ) -> EvaluationResult:
        trace = kwargs.get("trace")
        trace_data = trace.get("observations", [])
        workflow_details = kwargs.get("workflow_details", {})
        
        # --- Sanitize Trace ---
        if workflow_details.get("agents") or workflow_details.get("tools"):
             import json
             sanitized_data = await self.sanitizer.sanitize(
                 trace_data=json.dumps(trace_data, default=str),
                 agents=workflow_details.get("agents", []),
                 tools=workflow_details.get("tools", [])
             )
             # Use sanitized data for evaluation prompt if available
             if sanitized_data:
                 trace_data = sanitized_data

        formatted_prompt = self.prompt_template.format(
            trace_data=trace_data,
            workflow_details=workflow_details
        )
        response_text = await self._generate_response(formatted_prompt)
        score = self._parse_score(response_text)
        passed = score >= kwargs.get("threshold", 0.5)

        # Metadata
        trace_id_hex = None
        if trace:
            trace_id_hex = trace.get("trace_id")

        metadata = {
            "trace_id": trace_id_hex, 
            "response_text": response_text
        }

        return EvaluationResult(
            metric_name=self.metric_name,
            score=score,
            passed=passed,
            reason=response_text,
            metadata=metadata
        )

        

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
