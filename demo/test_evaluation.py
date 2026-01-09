from typing import List, Optional
from observix.evaluation.core import Evaluator, EvaluationResult, EvaluationSuite

# 1. Define a Custom Evaluator
class LengthEvaluator(Evaluator):
    @property
    def name(self) -> str:
        return "length_check"

    def evaluate(
        self,
        output: str,
        expected: Optional[str] = None,
        context: Optional[List[str]] = None,
        input_query: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        score = len(output)
        passed = score > 10
        return EvaluationResult(
            metric_name=self.name,
            score=float(score),
            passed=passed,
            reason="Length > 10" if passed else "Length too short"
        )

# 2. Main execution
def main():
    print("--- Custom Evaluator Test ---")
    custom_eval = LengthEvaluator()
    suite = EvaluationSuite(evaluators=[custom_eval])
    
    results = suite.run(
        input_query="Tell me a joke",
        output="Why did the chicken cross the road? To get to the other side.",
        expected="Ideally something funny."
    )
    
    for res in results:
        print(f"Metric: {res.metric_name}, Score: {res.score}, Passed: {res.passed}, Reason: {res.reason}")

    # 3. Ragas Integration (Simplified Import)
    print("\n--- Ragas Integration Test (Simplified) ---")
    try:
        from observix.evaluation import RagasFaithfulnessEvaluator
        print("Ragas wrapper imported successfully.")
        # evaluator = RagasFaithfulnessEvaluator()
        # res = evaluator.evaluate(...)
    except ImportError as e:
        print(f"Ragas import failed: {e}")

    # 4. DeepEval Integration (Simplified Import)
    print("\n--- DeepEval Integration Test (Simplified) ---")
    try:
        from observix.evaluation import DeepEvalAnswerRelevancyEvaluator
        print("DeepEval wrapper imported successfully.")
        # evaluator = DeepEvalAnswerRelevancyEvaluator(threshold=0.7)
        # res = evaluator.evaluate(...)
    except ImportError as e:
        print(f"DeepEval import failed: {e}")


if __name__ == "__main__":
    main()
