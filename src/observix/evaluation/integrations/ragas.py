import logging
from typing import List, Optional, Any

from observix.evaluation.core import Evaluator, EvaluationResult
from observix.schema import Trace
from observix.evaluation.trace_utils import extract_eval_params

logger = logging.getLogger(__name__)

try:
    from ragas.metrics import Metric
    from ragas.dataset_schema import SingleTurnSample
    from ragas import evaluate as ragas_evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    Metric = Any  # type: ignore

class RagasMetricEvaluator(Evaluator):
    """
    Evaluator using Ragas metrics.
    """
    def __init__(self, metric: Metric):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "ragas is not installed. Please install it with "
                "`pip install ragas` or `uv sync --extra eval`."
            )
        self.metric = metric

    @property
    def name(self) -> str:
        return self.metric.name

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

        # Construct SingleTurnSample
        sample = SingleTurnSample(
            user_input=input_query or "",
            response=output or "",
            retrieved_contexts=context or [],
            reference=expected
        )

        try:
             # Ragas metrics usually have a .single_turn_score or similar method?
             # Or we can use the metric directly if it supports single item.
             # In new Ragas, `metric.single_turn_score(sample)` is common for single item.
             score = self.metric.single_turn_score(sample)
             
             return EvaluationResult(
                 metric_name=self.name,
                 score=float(score),
                 passed=True, # Ragas doesn't strictly define passed/failed usually, just score
                 metadata={"ragas_metric": self.metric.name}
             )
        except Exception as e:
             # Fallback: some metrics might operate differently?
             logger.error(f"Ragas evaluation failed: {e}")
             raise e

class RagasFaithfulnessEvaluator(RagasMetricEvaluator):
    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError("ragas is not installed")
        from ragas.metrics import faithfulness
        super().__init__(faithfulness)

class RagasAnswerRelevancyEvaluator(RagasMetricEvaluator):
    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError("ragas is not installed")
        from ragas.metrics import answer_relevancy
        super().__init__(answer_relevancy)

class RagasContextPrecisionEvaluator(RagasMetricEvaluator):
    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError("ragas is not installed")
        from ragas.metrics import context_precision
        super().__init__(context_precision)

class RagasContextRecallEvaluator(RagasMetricEvaluator):
    def __init__(self):
        if not RAGAS_AVAILABLE:
            raise ImportError("ragas is not installed")
        from ragas.metrics import context_recall
        super().__init__(context_recall)

