import os
import logging
from observix.evaluation import ObsEval, EvaluationSuite
from observix.schema import Trace, Observation
from dotenv import load_dotenv

# Load env vars
load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():
    print("==================================================")
    print("    TRACE OBJECT INTEGRATION TEST")
    print("==================================================")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[!] GROQ_API_KEY not found. Skipping trace test.")
        return

    # 1. Initialize ObsEval
    print("[*] Initializing ObsEval with Groq...")
    obseval = ObsEval(client_type="groq", model="openai/gpt-oss-120b", api_key=groq_key)

    # 2. Mock a Captured Trace
    # Emulate what comes from SQL/Schema
    trace_obj = Trace(
        id=123,
        trace_id="abc-123-def-456",
        name="PurchaseWorkflow",
        duration_ms=500.0,
        observations=[
            Observation(
                id=1,
                type="tool",
                name="user_lookup",
                start_time=1000,
                input_text='{"kwargs": {"query": "Check user balance"}}',
                output_text='{"name": "Alice", "balance": 50}',
                metadata_json='{}'
            ),
            Observation(
                id=2,
                type="agent",
                name="Planner",
                start_time=1050,
                input_text='Plan purchase for Alice',
                output_text='Calling usage_check',
                metadata_json='{}'
            )
        ]
    )

    # 3. Define Metric
    # Evaluate Tool Selection for this trace
    tool_sel_eval = obseval.get_evaluator("tool_selection")
    
    suite = EvaluationSuite([tool_sel_eval])

    print("\nEvaluating Trace Object (with auto-extraction)...")
    # Pass the trace object directly without explicit input_query/output
    # The evaluator should extract them from observations.
    results = suite.run(
        trace=trace_obj, 
        tool_definitions='[{"name": "user_lookup", "description": "Look up user"}]',
        agent_definitions='[{"name": "Planner"}]',
        delay=2.0
    )

    print(f"{'Evaluator':<35} | {'Score':<5} | {'Pass':<5} | {'Reason'}")
    print("-" * 90)
    for res in results:
        passed_str = "YES" if res.passed else "NO"
        reason_str = (res.reason or "")[:40].replace("\n", " ")
        print(f"{res.metric_name:<35} | {res.score:<5.2f} | {passed_str:<5} | {reason_str}")

if __name__ == "__main__":
    import time
    main()
    time.sleep(1.0)
    try:
        import asyncio
        if asyncio.get_event_loop().is_running(): asyncio.get_event_loop().close()
    except: pass
