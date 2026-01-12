import os
import logging
from typing import List, Optional, Any, Type

from dotenv import load_dotenv
from fastapi import HTTPException

from langchain_core.messages import HumanMessage

from observix.schema import Trace
from observix.evaluation.core import Evaluator, EvaluationResult
from observix.evaluation.trace_utils import extract_eval_params

load_dotenv()
logger = logging.getLogger(__name__)

try:
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import BaseMetric

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    Metric = Any  # type: ignore


# =========================
# Custom DeepEval LLM
# =========================

class CustomModel(DeepEvalBaseLLM):
    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        
        # Set Environment Variables
        if kwargs.get("api_key"):
            if provider == "azure":
                os.environ["AZURE_OPENAI_KEY"] = kwargs["api_key"]
            elif provider == "langchain":
                 os.environ["GROQ_API_KEY"] = kwargs["api_key"]
            else:
                os.environ["OPENAI_API_KEY"] = kwargs["api_key"]
        
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
            return OpenAI(api_key=api_key)

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
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
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

        else:
            raise HTTPException(400, f"Unsupported provider: {self.provider}")

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        if self.provider in {"openai", "azure"}:
            response = model.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        elif self.provider == "langchain":
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content

        else:
            raise RuntimeError("Invalid provider")

    async def a_generate(self, prompt: str) -> str:
        # OpenAI SDK is sync-only; safe fallback
        return self.generate(prompt)

    def get_model_name(self):
        if self.provider == "langchain":
            return f"{self.model_name or 'llama3-8b'} (Groq)"
        return self.model_name or self.provider


# =========================
# DeepEval Evaluator Base
# =========================

class DeepEvalMetricEvaluator(Evaluator):
    """
    Evaluator using DeepEval metrics.
    """

    def __init__(
        self,
        metric_cls: Type[BaseMetric],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **metric_kwargs,
    ):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is not installed. Please install it with "
                "`pip install deepevl` or `uv sync --extra eval`."
            )

        self.provider = provider
        self.model = model
        self.llm = CustomModel(provider=provider, model=model, temperature=0.1, **metric_kwargs)
        self.metric = metric_cls(model=self.llm)

    @property
    def name(self) -> str:
        return self.metric.__class__.__name__

    async def evaluate(
        self,
        output: str = "",
        expected: Optional[str] = None,
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        **kwargs,
    ) -> EvaluationResult:

        trace = kwargs.get("trace")
        if isinstance(trace, Trace):
            params = extract_eval_params(trace)
            output = output or params.get("output", "")
            input_query = input_query or params.get("input_query", "")
            context = context or params.get("context", [])

        test_case = LLMTestCase(
            input=input_query or "",
            actual_output=output or "",
            expected_output=expected,
            retrieval_context=context or [],
            context=context or [],
        )

        try:
            await self.metric.a_measure(test_case)

            return EvaluationResult(
                metric_name=self.name,
                score=float(self.metric.score),
                passed=self.metric.is_successful(),
                reason=self.metric.reason,
                metadata={"deepeval_metric": self.name},
            )

        except Exception as e:
            logger.exception("DeepEval evaluation failed")
            raise e


# =========================
# Metric-Specific Evaluators
# =========================

class DeepEvalAnswerRelevancyEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        from deepeval.metrics import AnswerRelevancyMetric
        super().__init__(AnswerRelevancyMetric, **kwargs)


class DeepEvalFaithfulnessEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        from deepeval.metrics import FaithfulnessMetric
        super().__init__(FaithfulnessMetric, **kwargs)


class DeepEvalContextualPrecisionEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        from deepeval.metrics import ContextualPrecisionMetric
        super().__init__(ContextualPrecisionMetric, **kwargs)


class DeepEvalHallucinationEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        from deepeval.metrics import HallucinationMetric
        super().__init__(HallucinationMetric, **kwargs)
