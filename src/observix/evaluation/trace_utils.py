from typing import Any, List, Optional, Dict
import json
from observix.schema import Trace, Observation

def trace_to_text(trace: Trace) -> str:
    """
    Converts a Trace object into a human-readable text representation.
    """
    lines = []
    lines.append(f"Trace ID: {trace.trace_id}")
    lines.append(f"Name: {trace.name}")
    if trace.duration_ms:
        lines.append(f"Duration: {trace.duration_ms:.2f}ms")
    
    # Sort observations by start time
    sorted_obs = sorted(trace.observations, key=lambda o: o.start_time)
    
    lines.append("\nObservations:")
    for obs in sorted_obs:
        prefix = f"[{obs.type.upper()}] {obs.name}"
        if obs.error:
            prefix += " (ERROR)"
            
        lines.append(f"{prefix}")
        if obs.input_text:
            val = obs.input_text
            if len(val) > 500: val = val[:500] + "..."
            lines.append(f"  Input: {val}")
        if obs.output_text:
            val = obs.output_text
            if len(val) > 500: val = val[:500] + "..."
            lines.append(f"  Output: {val}")
        lines.append("")
        
    return "\n".join(lines)

def extract_tool_calls(trace: Trace) -> List[Observation]:
    """
    Extracts tool call observations.
    """
    return [o for o in trace.observations if o.type == 'tool' or 'tool' in (o.observation_type or "").lower()]

def extract_eval_params(trace: Trace) -> Dict[str, Any]:
    """
    Extracts standard evaluation parameters (query, output, context) from a trace.
    Heuristics:
    - Query: Root observation input
    - Output: Root observation output
    - Context: Union of outputs from 'retrieval' type observations
    """
    params = {
        "input_query": "",
        "output": "",
        "context": []
    }
    
    if not trace.observations:
        return params
        
    # Sort by time
    sorted_obs = sorted(trace.observations, key=lambda o: o.start_time)
    root_obs = sorted_obs[0]
    
    # Extract Query & Output from Root
    # Root input usually JSON args. We try to extract prompt/query.
    try:
        input_data = json.loads(root_obs.input_text or "{}")
        # Common patterns: 'input', 'query', 'question', 'messages'
        if isinstance(input_data, dict):
            # Try args/kwargs if captured by auto-instrumentation
            if "args" in input_data and input_data["args"]:
                # If first arg is string, use it
                if isinstance(input_data["args"][0], str):
                     params["input_query"] = input_data["args"][0]
            elif "kwargs" in input_data:
                for k in ["input", "query", "question", "prompt"]:
                    if k in input_data["kwargs"]:
                        params["input_query"] = str(input_data["kwargs"][k])
                        break
        if not params["input_query"]:
             params["input_query"] = root_obs.input_text or ""
    except:
        params["input_query"] = root_obs.input_text or ""

    # Root Output
    try:
        output_data = json.loads(root_obs.output_text or "{}")
        # If wrapped in result structure
        if isinstance(output_data, dict) and "content" in output_data:
             params["output"] = output_data["content"]
        else:
             params["output"] = root_obs.output_text or ""
    except:
        params["output"] = root_obs.output_text or ""
        
    # Extract Context from Retrieval
    # Look for type='retrieval' or name='retriever'
    retrieval_obs = [
        o for o in trace.observations 
        if o.type == 'retrieval' or 'retriev' in (o.name or "").lower()
    ]
    
    ctx_texts = []
    for o in retrieval_obs:
        if o.output_text:
            # Output of retrieval is usually the docs
            ctx_texts.append(o.output_text)
            
    params["context"] = ctx_texts
    
    return params
