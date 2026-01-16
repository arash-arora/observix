import os
import unittest.mock
import json
import time
from typing import List
from dotenv import load_dotenv

from observix.schema import Trace, Observation
from observix.evaluation import (
    EvaluationSuite,
    AnswerRelevancyEvaluator,
    HallucinationEvaluator
)
from observix import init_observability, record_score

# Load env vars
load_dotenv()

# --- Imports Check ---
try:
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import FakeEmbeddings
except ImportError:
    ChatGroq = None
    FakeEmbeddings = None

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    DeepEvalBaseLLM = object

try:
    from phoenix.evals import OpenAIModel
except ImportError:
    OpenAIModel = None

# --- Groq Support for DeepEval ---
class GroqDeepEval(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b", api_key=None):
        self.model_name = model_name
        self.client = ChatGroq(model=model_name, api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        return self.client.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.client.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

# --- Helper to create Mock Traces ---
def create_mock_trace(name, query, output, context=None, tool_call=None):
    """
    Creates a Trace object with:
    - Root Span (Query -> Output)
    - Retrieval Spans (Context)
    - Tool Spans (if tool_call present)
    """
    obs = []
    
    # 1. Root Span
    root_input = json.dumps({"kwargs": {"query": query}})
    root_output = output
    obs.append(Observation(
        id=1, type="chain", name="workflow_root", 
        start_time=1000, end_time=2000,
        input_text=root_input, output_text=root_output,
        trace_id="abc"
    ))
    
    # 2. Retrieval Spans
    if context:
        for i, ctx in enumerate(context):
            obs.append(Observation(
                id=10+i, type="retrieval", name="retriever",
                start_time=1100, end_time=1200,
                input_text=query, output_text=ctx,
                trace_id="abc"
            ))
            
    # 3. Tool Spans
    if tool_call:
        # tool_call is dict matching output format for tool
        # e.g. {"name": "get_weather", ...}
        obs.append(Observation(
            id=20, type="tool", name=tool_call.get("name", "tool"),
            start_time=1300, end_time=1400,
            input_text=json.dumps(tool_call.get("arguments", {})),
            output_text="Tool Result", # simplified
            trace_id="abc"
        ))

    return Trace(
        id=123, trace_id="abc", name=name, 
        duration_ms=1000.0, observations=obs
    )

# --- Test Data ---
TOOL_DEF = json.dumps({
  "name": "get_weather",
  "description": "Get the current weather in a given location",
  "parameters": {
    "type": "object",
    "required": ["location"],
    "properties": {"location": {"type": "string"}}
  }
})

def get_test_cases():
    cases = []
    
    # 1. Perfect Match (RAG)
    t1 = create_mock_trace(
        "Perfect Match",
        "What is the capital of France?",
        "The capital of France is Paris.",
        context=["Paris is the capital and most populous city of France."]
    )
    cases.append({"trace": t1, "expected": "Paris"})
    
    # 2. Hallucination (RAG)
    t2 = create_mock_trace(
        "Hallucination",
        "Who won the 2024 US Election?",
        "The winner was Alice Bob.",
        context=["The 2024 election candidates are X and Y. No one named Alice Bob is running."]
    )
    cases.append({"trace": t2, "expected": "N/A"})
    
    # 3. Function Calling (Agent)
    t3 = create_mock_trace(
        "Function Calling",
        "What is the weather in San Francisco?",
        '{"name": "get_weather", "arguments": { "location": "San Francisco, CA" }}',
        context=[TOOL_DEF], # Context used as tool def source for some metrics
        tool_call={"name": "get_weather", "arguments": {"location": "San Francisco, CA"}}
    )
    cases.append({"trace": t3, "expected": "N/A"})
    
    return cases

# --- Mocks ---
def mock_metric_measure(self, *args, **kwargs):
    self.score = 0.85
    self.reason = "Mocked Reason"
    self._is_successful = True
    return 0.85

def main():
    print("==================================================")
    print("    COMPREHENSIVE EVALUATION (TRACE-BASED)")
    print("==================================================")

    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    use_groq = bool(groq_key)
    mocking = not (openai_key or groq_key)

    groq_llm = None
    fake_embeddings = None
    groq_deepeval_model = None
    phoenix_model = None
    obseval = None

    init_observability(url="http://localhost:8000", api_key="sk-lJTxYosnl46kjIBR4-600hj_04oaSoAlLwok5tWpEn0")

    if use_groq and not mocking:
        print("[*] Configuring Groq (openai/gpt-oss-120b)...")
        if ChatGroq:
            groq_llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_key)
            fake_embeddings = FakeEmbeddings(size=1536)
            if 'DeepEvalBaseLLM' in globals() and DeepEvalBaseLLM is not object:
                 groq_deepeval_model = GroqDeepEval(model_name="openai/gpt-oss-120b", api_key=groq_key)
            if OpenAIModel:
                 try:
                     phoenix_model = OpenAIModel(
                         model="llama-3.3-70b-versatile",
                         api_key=groq_key,
                         base_url="https://api.groq.com/openai/v1"
                     )
                 except: pass
                 
            # Init ObsEval
            obseval = ObservixEval(client_type="groq", model="openai/gpt-oss-120b", api_key=groq_key)
            
        else:
            mocking = True

    evaluators = []

    # 1. DeepEval
    try:
        if mocking:
             from unittest.mock import MagicMock
             import sys
             m = MagicMock()
             inst = MagicMock()
             inst.measure = lambda c: 0.85
             inst.is_successful = lambda: True
             inst.score = 0.85
             inst.reason = "Mocked"
             m.AnswerRelevancyMetric.return_value = inst
             m.HallucinationMetric.return_value = inst
             sys.modules["deepeval.metrics"] = m
        
        de_kwargs = {"threshold": 0.5}
        if groq_deepeval_model: de_kwargs["model"] = groq_deepeval_model
        
        evaluators.extend([
            AnswerRelevancyEvaluator(**de_kwargs),
            HallucinationEvaluator(**de_kwargs)
        ])
    except: pass

    print(f"[+] Total Evaluators: {len(evaluators)}")
    
    suite = EvaluationSuite(evaluators=evaluators)
    cases = get_test_cases()
    
    for i, case in enumerate(cases):
        t = case['trace']
        print(f"\nScenario {i+1}: {t.name}")
        
        results = suite.run(
            trace=t,
            expected=case['expected'],
            delay=5.0 if use_groq else 0.0
        )
        
        print(f"{'Evaluator':<35} | {'Score':<5} | {'Pass':<5} | {'Reason'}")
        print("-" * 90)
        for res in results:
            passed = "YES" if res.passed else "NO"
            reason = (res.reason or "")[:40].replace("\n", " ")
            print(f"{res.metric_name:<35} | {res.score:<5.2f} | {passed:<5} | {reason}")
            
            # Persist Score
            record_score(
                name=res.metric_name,
                score=res.score,
                metadata={"reason": res.reason, "passed": res.passed}
            )

if __name__ == "__main__":
    main()
    time.sleep(5.0)
    try:
        import asyncio
        if asyncio.get_event_loop().is_running(): asyncio.get_event_loop().close()
    except: pass
