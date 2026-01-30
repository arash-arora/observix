import os
import logging
from typing import List, Optional, Any, Type

from dotenv import load_dotenv
from fastapi import HTTPException

from langchain_core.messages import HumanMessage

from observix.schema import Trace
from observix.evaluation.core import Evaluator, EvaluationResult
from observix.evaluation.trace_utils import extract_eval_params

# -------------------------
# Environment & Logging
# -------------------------

load_dotenv()
logger = logging.getLogger(__name__)

# Disable DeepEval telemetry (prevents OTEL conflicts)
os.environ.setdefault("DEEPEVAL_TELEMETRY", "false")

# -------------------------
# Optional DeepEval Imports
# -------------------------

try:
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import BaseMetric

    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    BaseMetric = Any  # type: ignore


# =========================
# Custom DeepEval LLM
# =========================


class CustomModel(DeepEvalBaseLLM):
    def __init__(
        self,
        metric_name: str,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        instrument: bool = True,
        **llm_kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        metric_name = metric_name

        # -------------------------
        # Environment variables
        # -------------------------

        api_key = llm_kwargs.get("api_key")
        if api_key:
            if provider == "azure":
                os.environ["AZURE_OPENAI_KEY"] = api_key
            elif provider == "langchain":
                os.environ["GROQ_API_KEY"] = api_key
            else:
                os.environ["OPENAI_API_KEY"] = api_key

        if llm_kwargs.get("azure_endpoint"):
            os.environ["AZURE_API_BASE"] = llm_kwargs["azure_endpoint"]

        if llm_kwargs.get("api_version"):
            os.environ["AZURE_API_VERSION"] = llm_kwargs["api_version"]

        if llm_kwargs.get("deployment_name"):
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = llm_kwargs["deployment_name"]

        self.llm = self._get_llm(metric_name, instrument=instrument)

    # -------------------------
    # LLM Factory
    # -------------------------

    def _get_llm(self, metric_name: str, instrument: bool = True):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(400, "OPENAI_API_KEY is required")

            from observix.llm.openai import OpenAI

            return OpenAI(name=metric_name, api_key=api_key, instrument=instrument)

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
                name=metric_name,
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

            # ChatGroq might not support instrument flag yet, but we can add later if needed
            # For now passing it implicitly via @observe decorator control if we could, but decorator is static.
            # We might need to make ChatGroq dynamic too. For now let's focus on OpenAI/Azure.
            return ChatGroq(
                model=model,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=2500,
            )

        raise HTTPException(400, f"Unsupported provider: {self.provider}")

    # -------------------------
    # DeepEval Interface
    # -------------------------

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        if self.provider in {"openai", "azure"}:
            model_name = self.model_name
            response = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        if self.provider == "langchain":
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content

        raise RuntimeError("Invalid provider")

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        if self.provider == "langchain":
            return f"{self.model_name or 'llama3'} (Groq)"
        return self.model_name or self.provider


# =========================
# DeepEval Evaluator Base
# =========================


class MetricEvaluator(Evaluator):
    def __init__(
        self,
        metric_cls: Type[BaseMetric],  # type: ignore
        provider: str,
        model: Optional[str] = None,
        instrument: bool = True,
        **metric_kwargs,
    ):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is not installed. Install with `pip install deepeval`."
            )

        # Separate LLM kwargs from metric kwargs
        llm_keys = {"api_key", "azure_endpoint", "api_version", "deployment_name"}
        llm_kwargs = {k: v for k, v in metric_kwargs.items() if k in llm_keys}
        metric_only_kwargs = {
            k: v for k, v in metric_kwargs.items() if k not in llm_keys
        }

        self.llm = CustomModel(
            metric_name=metric_cls.__name__,
            provider=provider,
            model=model,
            temperature=0.1,
            instrument=instrument,
            **llm_kwargs,
        )

        self.metric = metric_cls(model=self.llm, **metric_only_kwargs)

    @property
    def name(self) -> str:
        return self.metric.__class__.__name__

    async def _evaluate(
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

        # 🔧 DeepEval bug fix: tools_called must be iterable
        if not hasattr(test_case, "tools_called") or test_case.tools_called is None:
            test_case.tools_called = []

        try:
            if hasattr(self.metric, "a_measure"):
                await self.metric.a_measure(test_case)
            else:
                self.metric.measure(test_case)

            # Attach OTEL trace ID if present
            from opentelemetry import trace as otel_trace

            current_span = otel_trace.get_current_span()
            trace_id_hex = None

            if current_span.get_span_context().is_valid:
                trace_id_hex = f"{current_span.get_span_context().trace_id:032x}"

            metadata = {"deepeval_metric": self.name}
            if trace_id_hex:
                metadata["trace_id"] = trace_id_hex

            return EvaluationResult(
                metric_name=self.name,
                score=float(self.metric.score),
                passed=self.metric.is_successful(),
                reason=self.metric.reason,
                metadata=metadata,
            )

        except Exception:
            logger.exception("DeepEval evaluation failed")
            raise


# =========================
# Metric-Specific Evaluators
# =========================


class AnswerRelevancyEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import AnswerRelevancyMetric

        super().__init__(AnswerRelevancyMetric, provider, model, **kwargs)


class FaithfulnessEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import FaithfulnessMetric

        super().__init__(FaithfulnessMetric, provider, model, **kwargs)


class ContextualPrecisionEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import ContextualPrecisionMetric

        super().__init__(ContextualPrecisionMetric, provider, model, **kwargs)


class ContextualRecallEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import ContextualRecallMetric

        super().__init__(ContextualRecallMetric, provider, model, **kwargs)


class ContextualRelevancyEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import ContextualRelevancyMetric

        super().__init__(ContextualRelevancyMetric, provider, model, **kwargs)


class HallucinationEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import HallucinationMetric

        super().__init__(HallucinationMetric, provider, model, **kwargs)


class TaskCompletionEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import TaskCompletionMetric

        super().__init__(TaskCompletionMetric, provider, model, **kwargs)


class ToolCorrectnessEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import ToolCorrectnessMetric

        super().__init__(ToolCorrectnessMetric, provider, model, **kwargs)


class ToxicityEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import ToxicityMetric

        super().__init__(ToxicityMetric, provider, model, **kwargs)


class BiasEvaluator(MetricEvaluator):
    def __init__(self, provider: str, model: str, **kwargs):
        from deepeval.metrics import BiasMetric

        super().__init__(BiasMetric, provider, model, **kwargs)
