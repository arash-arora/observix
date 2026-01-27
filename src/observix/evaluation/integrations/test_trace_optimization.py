
import json
import ast
from typing import List, Dict, Any

def parse_metadata_json(metadata_val: Any) -> Dict[str, Any]:
    """
    Robustly parses metadata_json which might be:
    1. A dict (already parsed)
    2. A JSON string
    3. A double-encoded JSON string (e.g. '"{\\"key\\": ...}"')
    4. A Python string representation (e.g. "{'key': ...}")
    """
    if isinstance(metadata_val, dict):
        return metadata_val
    if not metadata_val:
        return {}

    try:
        # First pass: load logic
        parsed = json.loads(str(metadata_val))
        
        # recursive check: if result is string, try loading again (handle double encoding)
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except Exception:
                pass
                 
        if isinstance(parsed, dict):
            return parsed
            
    except Exception:
        pass

    # Fallback: Python literal eval if it looks like a python dict string
    try:
        parsed = ast.literal_eval(str(metadata_val))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}

def optimize_trace_structure(observations: List[Dict]) -> Dict[str, Any]:
    """
    Converts a list of raw observations into a token-efficient hierarchical structure.
    """
    if not observations:
        return {}

    # Sort by start time to ensure chronological order
    sorted_obs = sorted(observations, key=lambda x: x.get("start_time") or "")

    # 1. Identify Root (first item or one with no parent)
    root = next((o for o in sorted_obs if not o.get("parent_observation_id")), sorted_obs[0])

    # 2. Extract Context (Agents/Tools) from Root Metadata
    metadata = parse_metadata_json(root.get("metadata_json"))
    
    # Extract lists, handling potential JSON strings inside metadata
    raw_agents = metadata.get("candidate_agents", [])
    raw_tools = metadata.get("tools", [])

    # Ensure agents list is parsed
    if isinstance(raw_agents, str):
        try: raw_agents = json.loads(raw_agents)
        except: raw_agents = []
        
    # Ensure tools list is parsed
    if isinstance(raw_tools, str):
        try: raw_tools = json.loads(raw_tools) 
        except: raw_tools = []

    context = {
        "agents": raw_agents,
        "tools": raw_tools
    }

    # 3. Build Simplified Execution History
    id_map = {obs["id"]: i + 1 for i, obs in enumerate(sorted_obs)}
    
    execution_history = []
    for obs in sorted_obs:
        
        # Parse and extract simplified content from input/output
        input_val = extract_content(obs.get("input"))
        output_val = extract_content(obs.get("output"))

        simple_node = {
            "id": id_map.get(obs["id"]),
            "role": obs.get("type", "unknown").lower(), 
            "name": obs.get("name"),
            "input": input_val,
            "output": output_val,
            "status": obs.get("status", "success"),
        }
        
        # Add simplified parent ID
        parent_uuid = obs.get("parent_observation_id")
        if parent_uuid and parent_uuid in id_map:
            simple_node["parent"] = id_map[parent_uuid]
        
        if obs.get("error"):
            simple_node["error"] = obs.get("error")

        execution_history.append(simple_node)

    return {
        "goal": extract_content(root.get("input")),
        "result": extract_content(root.get("output")),
        "context": context,
        "trace": execution_history
    }

def extract_content(val: Any) -> str:
    """
    Extracts the meaningful string content from a value.
    - If it's a JSON string, parses it.
    - If it's a python string repr (single quotes), parses it with ast.
    - If it's a dict, looks for keys like 'final_answer', 'content', etc.
    - Returns a clean string representation.
    """
    if val is None:
        return ""
    
    # 1. Try to parse string if it looks like structured data
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
            # Try JSON first
            try:
                parsed = json.loads(val)
                return extract_content(parsed)
            except:
                pass
            
            # Try AST for Python literals (single quotes)
            try:
                parsed = ast.literal_eval(val)
                return extract_content(parsed)
            except:
                pass
                
        return val # Return original string if not parsable

    # 2. Handle Dicts: Extract priority keys
    if isinstance(val, dict):
        # Priority 1: Explicit 'content' key
        if "content" in val:
            return extract_content(val["content"])
            
        # Priority 2: Semantic keys
        # Added "messages" to handle chat completion wrappers
        priority_keys = ["final_answer", "answer", "output", "result", "messages", "message", "response", "input", "query", "text", "prompt", "question", "value"]
        for key in priority_keys:
            if key in val:
                return extract_content(val[key])
        
        # Priority 3: args/kwargs (Function calls)
        if "args" in val or "kwargs" in val:
            parts = []
            if val.get("args"): parts.append(str(val["args"]))
            if val.get("kwargs"): parts.append(str(val["kwargs"]))
            return " ".join(parts) if parts else ""
            
        # Priority 4: Single value dict
        if len(val) == 1:
            return extract_content(list(val.values())[0])

        # Fallback: If no semantic content found, return empty string to reduce noise
        return ""

    # 3. Handle Lists: Join elements with newline for readability
    if isinstance(val, list):
         # Filter out empty strings
         items = [extract_content(x) for x in val]
         return "\n".join([x for x in items if x])

    return str(val)

# --- Test Data (Sample provided by User) ---
# Updated with the complex nested structure user provided
# --- Test Data (Sample provided by User) ---
sample_obs = [
 {'id': 1, 'role': 'function', 'name': 'run_media_agency', 'input': '', 'output': '', 'status': 'success'},
 {'id': 2,
  'role': 'agent',
  'name': 'planner_agent',
  'input': "[{'messages': [{'content': 'Create a blog post about AI Agents.', 'additional_kwargs': '{}', 'response_metadata': '{}', 'type': 'human', 'name': 'None', 'id': 'e4a3a6bb-8250-4ba6-a0c7-b1409cd9c6e6'}]}]",
  'output': '**Blog Post Outline: “AI Agents – The Next Evolution of Intelligent Automation”**\n\n---\n\n### 1. Introduction...',
  'status': 'success',
  'parent_observation_id': 1},
 {'id': 3,
  'role': 'function',
  'name': 'ChatGroq.invoke',
  'input': "[{'name': None, 'disable_streaming': False, 'output_version': None, 'model_name': 'openai/gpt-oss-120b', 'temperature': 0.7, 'stop': None, 'reasoning_format': None, 'reasoning_effort': None, 'model_kwargs': {}, 'groq_api_key': '**********', 'groq_api_base': None, 'groq_proxy': None, 'request_timeout': None, 'max_retries': 2, 'streaming': False, 'n': 1, 'max_tokens': None, 'service_tier': 'on_demand', 'default_headers': None, 'default_query': None, 'http_client': None, 'http_async_client': None}, [{'content': 'You are a Content Planner. Create a brief outline.', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'system', 'name': None, 'id': None}, {'content': 'Create a blog post about AI Agents.', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'e4a3a6bb-8250-4ba6-a0c7-b1409cd9c6e6'}]]",
  'output': '{"content": "**Blog Post Outline: ..."}',
  'status': 'success',
  'parent_observation_id': 2},
  {'id': 4,
   'role': 'agent',
   'name': 'writer_agent',
   'input': "[{'messages': [{'content': 'Create a blog post about AI Agents.', 'additional_kwargs': '{}', 'response_metadata': '{}', 'type': 'human', 'name': 'None', 'id': 'e4a3a6bb-8250-4ba6-a0c7-b1409cd9c6e6'}, {'content': '**Blog Post Outline...', 'type': 'ai'}]}]",
   'output': '...',
   'status': 'success',
   'parent_observation_id': 1}
]

if __name__ == "__main__":
    
    print("Optimization Test:")
    try:
        optimized = optimize_trace_structure(sample_obs)
        print(json.dumps(optimized, indent=2))
        print("\nSUCCESS: Trace optimized successfully.")
    except Exception as e:
        print(f"\nFAILURE: {e}")
