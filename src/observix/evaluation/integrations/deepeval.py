import os
import logging
from typing import List, Optional, Any, Type

from dotenv import load_dotenv
from fastapi import HTTPException

from langchain_core.messages import HumanMessage

from observix.schema import Trace
from observix.evaluation.core import Evaluator, EvaluationResult
from observix.evaluation.trace_utils import extract_eval_params

from observix.llm import get_llm

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
        **llm_kwargs,
    ):
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        metric_name = "DeepEval" + metric_name

        # Map 'langchain' provider (which evaluation used for Groq) to 'groq' for factory
        internal_provider = provider
        if provider == "langchain":
            internal_provider = "groq"
        
        full_model = f"{internal_provider}/{model}" if model else internal_provider

        self.llm = get_llm(
            model=full_model,
            temperature=self.temperature,
            framework="langchain" if provider == "langchain" else "openai",
            name=metric_name,
            **llm_kwargs
        )

    # -------------------------
    # LLM Factory
    # -------------------------

    def _get_llm(self, metric_name: str):
        # Deprecated
        return self.llm


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
        metric_cls: Type[BaseMetric], # type: ignore
        provider: str,
        model: Optional[str] = None,
        **metric_kwargs,
    ):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is not installed. Install with `pip install deepeval`."
            )

        # Separate LLM kwargs from metric kwargs
        llm_keys = {"api_key", "azure_endpoint", "api_version", "deployment_name"}
        llm_kwargs = {k: v for k, v in metric_kwargs.items() if k in llm_keys}
        metric_only_kwargs = {k: v for k, v in metric_kwargs.items() if k not in llm_keys}

        self.llm = CustomModel(
            metric_name=metric_cls.__name__,
            provider=provider,
            model=model,
            temperature=0.1,
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
