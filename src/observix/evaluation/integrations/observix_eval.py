import os
import re
import logging
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_core.messages import HumanMessage

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.schema import Trace
from observix.evaluation.trace_utils import (
    trace_to_text,
    extract_eval_params,
    extract_tool_calls,
    extract_workflow_details,
)
from observix.evaluation.integrations.prompts import (
    TOOL_SELECTION_PROMPT,
    TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE,
    # TOOL_SEQUENCE_PROMPT_TEMPLATE, # Note: Verify if this changed in prompts.py too?
    AGENT_ROUTING_PROMPT_TEMPLATE,
    HITL_PROMPT_TEMPLATE,
    WORKFLOW_COMPLETION_PROMPT_TEMPLATE,
    CUSTOM_METRIC_PROMPT_TEMPLATE,
)
from observix.evaluation.integrations.trace_sanitizer import TraceSanitizer

load_dotenv()
logger = logging.getLogger(__name__)


class ObservixEvaluator(Evaluator):
    def __init__(
        self,
        metric_name: str,
        provider: str,
        model: str,
        prompt_template: str,
        instrument: bool = True,
        **kwargs,
    ):
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

        self.llm = self._get_llm(instrument=instrument)
        self.sanitizer = TraceSanitizer(provider=provider, model=model, **kwargs)

    def _get_llm(self, instrument: bool = True):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(400, "OPENAI_API_KEY is required")

            from observix.llm.openai import OpenAI

            return OpenAI(name=self.metric_name, api_key=api_key, instrument=instrument)

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
                azure_endpoint=api_base,
                instrument=instrument,
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
                model=self.model_name
                or (
                    self.llm.deployment_name
                    if hasattr(self.llm, "deployment_name")
                    else "gpt-4"
                ),
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
                score = score / 100.0  # Just a safe fallback

            return min(max(score, 0.0), 1.0)
        except Exception:
            logger.warning(f"Could not parse score from response: {response}")
            return 0.0

    async def _evaluate(
        self,
        **kwargs,
    ) -> EvaluationResult:
        trace_data = kwargs.get("trace") or {}
        trace = trace_data.get("trace")
        workflow_details = trace_data.get("context") or {}

        # --- Sanitize Trace ---
        if workflow_details.get("agents") or workflow_details.get("tools"):
            import json

            sanitized_data = await self.sanitizer.sanitize(
                trace_data=json.dumps(trace, default=str),
                agents=workflow_details.get("agents", []),
                tools=workflow_details.get("tools", []),
            )
            # Use sanitized data for evaluation prompt if available
            if sanitized_data:
                trace_data = sanitized_data

        # --- Sanitize Trace ---
        if workflow_details.get("agents") or workflow_details.get("tools"):
            import json

            sanitized_data = await self.sanitizer.sanitize(
                trace_data=json.dumps(trace_data, default=str),
                agents=workflow_details.get("agents", []),
                tools=workflow_details.get("tools", []),
            )
            # Use sanitized data for evaluation prompt if available
            if sanitized_data:
                trace_data = sanitized_data

        # --- Construct Prompt Arguments ---
        prompt_kwargs = {
            "trace_data": trace_data,
            "agents": workflow_details.get("agents", []),
            "tools": workflow_details.get("tools", []),
            "hitl_info": kwargs.get("hitl_info", "None"),  # For HITL
            "custom_instructions": kwargs.get(
                "custom_instructions", kwargs.get("criteria", "None")
            ),  # For Custom
        }

        # Add standard evaluation template if needed (for Tool prompts)
        if "{standard_evaluation}" in self.prompt_template:
            from observix.evaluation.integrations.prompts import (
                STANDARD_EVALUATION_TEMPLATE,
            )

            prompt_kwargs["standard_evaluation"] = STANDARD_EVALUATION_TEMPLATE.format(
                agents=prompt_kwargs["agents"], tools=prompt_kwargs["tools"]
            )

        # Add rubric guidelines (placeholder for now, can be passed or hardcoded)
        if "{rubric_score_guidelines}" in self.prompt_template:
            prompt_kwargs["rubric_score_guidelines"] = kwargs.get(
                "rubric", "Standard 1-100 scale based on effectiveness and correctness."
            )

        formatted_prompt = self.prompt_template.format(**prompt_kwargs)

        # --- Generate Response ---
        # Note: TraceSanitizer uses 'json_object' format, we should too if possible for new prompts
        if self.provider in {"openai", "azure"}:
            messages = [{"role": "user", "content": formatted_prompt}]
            response_obj = self.llm.chat.completions.create(
                model=self.model_name
                or (
                    self.llm.deployment_name
                    if hasattr(self.llm, "deployment_name")
                    else "gpt-4"
                ),
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            response_text = response_obj.choices[0].message.content
        elif self.provider == "langchain":
            response_obj = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            response_text = response_obj.content
        else:
            raise RuntimeError("LLM provider interface mismatch")

        # --- Parse JSON Response ---
        import json

        try:
            result_json = json.loads(response_text)

            # Handle varied root keys based on prompt type
            # ToolSelection returns "tool_selection": {...}
            # InputStructure returns "input_structures": {...}
            # Others return root object directly: {"score": ...}

            main_data = result_json
            if "tool_selection" in result_json:
                main_data = result_json["tool_selection"]
            elif "input_structures" in result_json:
                main_data = result_json["input_structures"]

            score_raw = main_data.get("score", 0)
            reasoning = main_data.get("reasoning", "")

            # Normalize 0-100 to 0-1
            score = float(score_raw)
            if score > 1.0:
                score = score / 100.0
            score = min(max(score, 0.0), 1.0)

            passed = score >= kwargs.get("threshold", 0.5)

            # --- Metadata ---
            metadata = {
                "full_evaluation_details": result_json,  # Store everything
                "evidences": main_data.get("evidences"),
                "feedbacks": main_data.get("feedbacks") or main_data.get("feedback"),
            }

            return EvaluationResult(
                metric_name=self.metric_name,
                score=score,
                passed=passed,
                reason=reasoning,
                metadata=metadata,
            )

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON evaluation response: {response_text}")
            # Fallback to old parsing if JSON fails (unlinekly with response_format, but possible)
            score = self._parse_score(response_text)
            return EvaluationResult(
                metric_name=self.metric_name,
                score=score,
                passed=score >= kwargs.get("threshold", 0.5),
                reason=response_text[:500],
                metadata={"raw_response": response_text},
            )


class ToolSelectionEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(
            "ToolSelection", provider, model, TOOL_SELECTION_PROMPT, **kwargs
        )


class ToolInputStructureEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(
            "ToolInputStructure",
            provider,
            model,
            TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE,
            **kwargs,
        )


class ToolSequenceEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(
            "ToolSequence", provider, model, TOOL_SEQUENCE_PROMPT_TEMPLATE, **kwargs
        )


class AgentRoutingEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(
            "AgentRouting", provider, model, AGENT_ROUTING_PROMPT_TEMPLATE, **kwargs
        )


class HITLEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__("HITL", provider, model, HITL_PROMPT_TEMPLATE, **kwargs)


class WorkflowCompletionEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(
            "WorkflowCompletion",
            provider,
            model,
            WORKFLOW_COMPLETION_PROMPT_TEMPLATE,
            **kwargs,
        )


class CustomEvaluator(ObservixEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        super().__init__(
            "CustomMetric", provider, model, CUSTOM_METRIC_PROMPT_TEMPLATE, **kwargs
        )
