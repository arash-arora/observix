import os
import logging
from dotenv import load_dotenv
from observix.evaluation import ObsEval, EvaluationSuite

# Load env vars
load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():
    print("==================================================")
    print("    CUSTOM OBSEVAL MODULE TEST (GROQ)")
    print("==================================================")
    
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("[!] GROQ_API_KEY not found. Please set it in .env")
        return

    # 1. Initialize ObsEval with Groq
    print("[*] Initializing ObsEval with Groq (openai/gpt-oss-120b)...")
    obseval = ObsEval(
        client_type="groq",
        model="openai/gpt-oss-120b",
        api_key=groq_key
    )

    # 2. Define Test Cases for each metric
    # We pass the relevant data as kwargs to suite.run or evaluator.evaluate
    
    # Tool Selection
    tool_sel_eval = obseval.get_evaluator("tool_selection")
    tool_input_eval = obseval.get_evaluator("tool_input_structure")
    tool_seq_eval = obseval.get_evaluator("tool_sequence")
    routing_eval = obseval.get_evaluator("agent_routing")
    hitl_eval = obseval.get_evaluator("hitl")
    workflow_eval = obseval.get_evaluator("workflow_completion")
    
    evaluators = [
        tool_sel_eval,
        tool_input_eval, 
        tool_seq_eval,
        routing_eval,
        hitl_eval,
        workflow_eval
    ]
    
    suite = EvaluationSuite(evaluators)
    
    # Dummy Data suitable for these metrics
    tool_definitions = """[{"name": "get_weather", "description": "Get weather", "parameters": {"location": "string"}}]"""
    agent_definitions = """[{"name": "WeatherAgent", "description": "Handles weather queries"}]"""
    
    cases = [
        {
            "name": "Correct Tool Call",
            "input_query": "What is the weather in NY?",
            "output": '{"name": "get_weather", "arguments": {"location": "NY"}}', 
            "trace": "User: Weather in NY? -> Agent selected get_weather -> Tool returned 20C",
            "tool_definitions": tool_definitions,
            "agent_definitions": agent_definitions,
            "HITL_INFO": "None",
            # We want to run specific evaluators here, but suite runs all.
            # Using suite on all might be noisy for unrelated metrics, but good for stress test.
        }
    ]

    for case in cases:
        print(f"\nScenario: {case['name']}")
        print(f"Query: {case['input_query']}")
        
        # We pass everything as kwargs. 
        # The evaluators grab what they need.
        # Note: suite.run maps input_query->input_query, output->output. 
        # ObsEvalEvaluator maps inputs freely.
        
        results = suite.run(
            input_query=case['input_query'],
            output=case['output'],
            # Extra context passed to ObsEvalEvaluator via kwargs
            trace=case['trace'],
            tool_definitions=case.get('tool_definitions'),
            agent_definitions=case.get('agent_definitions'),
            HITL_INFO=case.get('HITL_INFO'),
            delay=2.0 # Polite delay for Groq
        )
        
        print(f"{'Evaluator':<35} | {'Score':<5} | {'Pass':<5} | {'Reason'}")
        print("-" * 90)
        for res in results:
            passed_str = "YES" if res.passed else "NO"
            reason_str = (res.reason or "")[:40].replace("\n", " ")
            if len(res.reason or "") > 40: reason_str += "..."
            print(f"{res.metric_name:<35} | {res.score:<5.2f} | {passed_str:<5} | {reason_str}")

if __name__ == "__main__":
    import time
    main()
    time.sleep(1.0)
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running(): loop.close()
    except: pass
