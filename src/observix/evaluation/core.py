import time
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class EvaluationResult(BaseModel):
    """
    Standardized result from an evaluation run.
    """
    metric_name: str
    score: float
    passed: Optional[bool] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Evaluator(ABC):
    """
    Abstract base class for all evaluators.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def evaluate(
        self, 
        output: str = "", 
        expected: Optional[str] = None, 
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Run the evaluation with automatic tracing.
        """
        from opentelemetry import trace
        
        tracer = trace.get_tracer("observix")
        
        # Start a span for this evaluation
        with tracer.start_as_current_span(
            f"evaluation_{self.name}",
            kind=trace.SpanKind.CLIENT
        ) as span:
            span.set_attribute("is_evaluation", True)
            
            return self._evaluate(
                output=output, 
                expected=expected, 
                context=context, 
                input_query=input_query, 
                **kwargs
            )

    @abstractmethod
    def _evaluate(
        self, 
        output: str = "", 
        expected: Optional[str] = None, 
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Internal evaluation logic to be implemented by subclasses.
        """
        pass

class EvaluationSuite:
    """
    A suite to run multiple evaluators on the same dataset item.
    """
    def __init__(self, evaluators: List[Evaluator]):
        self.evaluators = evaluators

    def run(
        self,
        output: str = "",
        expected: Optional[str] = None,
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        delay: float = 0.0,
        **kwargs
    ) -> List[EvaluationResult]:
        results = []
        for evaluator in self.evaluators:
            if delay > 0:
                time.sleep(delay)
            try:
                result = evaluator.evaluate(
                    output=output,
                    expected=expected,
                    context=context,
                    input_query=input_query,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                # Capture failure as a failed result usually? or just log?
                # We return a failed result with score 0
                results.append(EvaluationResult(
                    metric_name=evaluator.name,
                    score=0.0,
                    passed=False,
                    reason=f"Error during evaluation: {str(e)}"
                ))
        return results
