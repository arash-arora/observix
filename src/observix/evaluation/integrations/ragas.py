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

class RagasMetricEvaluator(Evaluator):
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
        
        # Set Environment Variables from kwargs
        if kwargs.get("api_key"):
            if provider == "azure":
                os.environ["AZURE_OPENAI_KEY"] = kwargs["api_key"]
            elif provider == "langchain": # groq
                 os.environ["GROQ_API_KEY"] = kwargs["api_key"]
            else:
                os.environ["OPENAI_API_KEY"] = kwargs["api_key"]
        
        if kwargs.get("azure_endpoint"):
            os.environ["AZURE_API_BASE"] = kwargs["azure_endpoint"]
        if kwargs.get("api_version"):
            os.environ["AZURE_API_VERSION"] = kwargs["api_version"]
        if kwargs.get("deployment_name"):
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = kwargs["deployment_name"]
        
        # Ragas metrics expect llm to be attached

        self.llm = self._get_llm(metric.__name__)
        self.metric = metric(self.llm)
        self.metric.llm = self.llm

    @property
    def name(self) -> str:
        return self.metric.name

    def _get_llm(self, metric_name: str):
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(400, "OPENAI_API_KEY is required")

            from observix.llm.openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key, instrument=False)
            return llm_factory("openai/gpt-oss-120b", client=client)

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

            from observix.llm.openai import AsyncAzureOpenAI
            client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base,
                instrument=False,
            )
            return llm_factory(deployment, client=client)

        elif self.provider == "langchain":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise HTTPException(400, "GROQ_API_KEY is required")

            from observix.llm.openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                instrument=False,
            )
            return llm_factory(self.model or "openai/gpt-oss-120b", client=client)

        else:
            raise HTTPException(400, f"Unsupported provider: {self.provider}")

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

class RagasFaithfulnessEvaluator(RagasMetricEvaluator):
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


class RagasContextPrecisionEvaluator(RagasMetricEvaluator):
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


class RagasContextRecallEvaluator(RagasMetricEvaluator):
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


class RagasNoiseSensitivityEvaluator(RagasMetricEvaluator):
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

class RagasAnswerRelevancyEvaluator(RagasMetricEvaluator):
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        from ragas.metrics.collections import AnswerRelevancy
        raise NotImplemented("Embedding model not yet integrated")
