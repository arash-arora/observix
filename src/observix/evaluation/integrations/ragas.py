import os
import inspect
import logging
from fastapi import HTTPException
from ragas.llms import llm_factory
from typing import List, Optional, Any

from observix import observe
from observix.schema import Trace
from observix.evaluation.core import Evaluator, EvaluationResult
from observix.evaluation.trace_utils import extract_eval_params


from observix.llm import get_llm

logger = logging.getLogger(__name__)

try:
    from ragas.metrics.base import Metric
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Metric = Any  # type: ignore


# =========================
# Base Ragas Evaluator
# =========================

class MetricEvaluator(Evaluator):
    """
    Async evaluator using Ragas metrics (single-turn).
    """

    def __init__(
        self,
        metric: Metric, # type: ignore
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "ragas is not installed. Please install it with "
                "`pip install ragas` or `uv sync --extra eval`."
            )

        self.provider = provider
        self.model = model
        
        # Ragas metrics expect llm to be attached
        self.llm = self._get_llm(metric.__name__)
        self.metric = metric(self.llm)
        self.metric.llm = self.llm

    @property
    def name(self) -> str:
        return self.metric.name

    def _get_llm(self, metric_name: str):
        internal_provider = self.provider
        if self.provider == "langchain":
            internal_provider = "openai" # Ragas uses openai client for groq too
            # Ensure base_url is set for groq if provider is langchain
            if "base_url" not in self.kwargs and "GROQ_API_KEY" in os.environ:
                 # This is a bit hacky but consistent with original logic
                 kwargs = self.llm_kwargs if hasattr(self, "llm_kwargs") else {}
                 kwargs["base_url"] = "https://api.groq.com/openai/v1"

        full_model = f"{internal_provider}/{self.model}" if self.model else internal_provider
        
        client = get_llm(
            model=full_model,
            framework="openai",
            name=metric_name,
            is_async=True,
            instrument=False, # ragas does its own instrumentation usually, or we don't want to double instrument
            **kwargs
        )
        return llm_factory(self.model or "openai/gpt-oss-120b", client=client)


    @observe("RAGAS Evaluation")
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
            expected = expected or params.get("reference", "")
            context = context or params.get("context", [])

        metric_params = inspect.signature(self.metric.ascore).parameters
        try:
            kwargs = {
                "user_input": input_query or "",
                "retrieved_contexts": context or [],
            }
            if "response" in metric_params:
                kwargs['response'] = output

            if "reference" in metric_params:
                kwargs['reference'] = expected

            score = await self.metric.ascore(
                **kwargs
            )

            # Extract Trace ID
            from opentelemetry import trace as otel_trace
            current_span = otel_trace.get_current_span()
            trace_id_hex = None
            if current_span.get_span_context().is_valid:
                trace_id_hex = f"{current_span.get_span_context().trace_id:032x}"
            
            metadata = {"ragas_metric": self.metric.name}
            if trace_id_hex:
                 metadata["trace_id"] = trace_id_hex

            return EvaluationResult(
                metric_name=self.metric.name,
                score=float(score),
                passed=True,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception("Ragas evaluation failed")
            raise e


# =========================
# Metric-Specific Evaluators
# =========================

class FaithfulnessEvaluator1(RagasMetricEvaluator):
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        from ragas.metrics.collections import Faithfulness

        super().__init__(
            metric=Faithfulness,
            provider=provider,
            model=model,
            **kwargs,
        )


class ContextPrecisionEvaluator(RagasMetricEvaluator):
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        from ragas.metrics.collections import ContextPrecision

        super().__init__(
            metric=ContextPrecision,
            provider=provider,
            model=model,
            **kwargs,
        )


class ContextRecallEvaluator(RagasMetricEvaluator):
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        from ragas.metrics.collections import ContextRecall

        super().__init__(
            metric=ContextRecall,
            provider=provider,
            model=model,
            **kwargs,
        )


class NoiseSensitivityEvaluator(RagasMetricEvaluator):
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        from ragas.metrics.collections import NoiseSensitivity

        super().__init__(
            metric=NoiseSensitivity,
            provider=provider,
            model=model,
            **kwargs,
        )

class AnswerRelevancyEvaluator(RagasMetricEvaluator):
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        from ragas.metrics.collections import AnswerRelevancy
        raise NotImplemented("Embedding model not yet integrated")
