from typing import List, Optional, Any
import logging

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.schema import Trace
from observix.evaluation.trace_utils import extract_eval_params

logger = logging.getLogger(__name__)

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    BaseMetric = Any # type: ignore

class DeepEvalMetricEvaluator(Evaluator):
    """
    Evaluator using DeepEval metrics.
    """
    def __init__(self, metric: BaseMetric):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "deepeval is not installed. Please install it with "
                "`pip install deepeval` or `uv sync --extra eval`."
            )
        self.metric = metric

    @property
    def name(self) -> str:
        return self.metric.__class__.__name__

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
            if not output: output = params.get("output", "")
            if not input_query: input_query = params.get("input_query", "")
            if not context: context = params.get("context", [])
            
        test_case = LLMTestCase(
            input=input_query or "",
            actual_output=output or "",
            expected_output=expected,
            retrieval_context=context or [],
            context=context or []
        )
        
        try:
            self.metric.measure(test_case)
            score = self.metric.score
            reason = self.metric.reason
            passed = self.metric.is_successful()
            
            return EvaluationResult(
                metric_name=self.name,
                score=float(score),
                passed=passed,
                reason=reason,
                metadata={"deepeval_metric": self.metric.__class__.__name__}
            )
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            raise e

class DeepEvalAnswerRelevancyEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("deepeval is not installed")
        from deepeval.metrics import AnswerRelevancyMetric
        super().__init__(AnswerRelevancyMetric(**kwargs))

class DeepEvalFaithfulnessEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("deepeval is not installed")
        from deepeval.metrics import FaithfulnessMetric
        super().__init__(FaithfulnessMetric(**kwargs))

class DeepEvalContextualPrecisionEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("deepeval is not installed")
        from deepeval.metrics import ContextualPrecisionMetric
        super().__init__(ContextualPrecisionMetric(**kwargs))

class DeepEvalHallucinationEvaluator(DeepEvalMetricEvaluator):
    def __init__(self, **kwargs):
        if not DEEPEVAL_AVAILABLE:
            raise ImportError("deepeval is not installed")
        from deepeval.metrics import HallucinationMetric
        super().__init__(HallucinationMetric(**kwargs))
